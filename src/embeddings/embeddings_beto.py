import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ─── Constantes ───────────────────────────────────────────────────────────────

BETO_MODEL = "bert-base-uncased"
MAX_LENGTH  = 512          # máximo de tokens que acepta BETO
BATCH_SIZE  = 16           # canciones por lote al generar embeddings


# ─── Cargador del modelo ──────────────────────────────────────────────────────

class CargadorBETO:
    """
    Singleton ligero para cargar tokenizer y modelo BETO una sola vez.

    Uso
    ---
        beto = CargadorBETO()
        tokenizer, model = beto.tokenizer, beto.model
    """

    def __init__(self, modelo: str = BETO_MODEL):
        print(f"Cargando BETO desde HuggingFace: {modelo}")
        print("(Puede tardar algunos minutos la primera vez...)")

        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.model     = AutoModel.from_pretrained(modelo)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parámetros  : {n_params:,}")
        print(f"  Dimensión   : {self.model.config.hidden_size}d")
        print(f"  Capas       : {self.model.config.num_hidden_layers}")
        print(f"  Atención    : {self.model.config.num_attention_heads} cabezas")
        print("BETO listo.\n")


# ─── Generación de embeddings ─────────────────────────────────────────────────

def embedding_cls(
    textos: List[str],
    tokenizer,
    model,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Genera embeddings usando el token [CLS] de la última capa de BETO.
    Procesa en lotes para no saturar la memoria.

    Parameters
    ----------
    textos      : lista de strings (letras de canciones)
    tokenizer   : tokenizer de BETO
    model       : modelo BETO en modo eval
    max_length  : truncar a este número de tokens
    batch_size  : tamaño del lote

    Returns
    -------
    np.ndarray de forma (n_textos, 768)
    """
    todos_embeddings = []

    for inicio in range(0, len(textos), batch_size):
        lote = textos[inicio:inicio + batch_size]
        inputs = tokenizer(
            lote,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # [CLS] → posición 0 de la secuencia
        cls_batch = outputs.last_hidden_state[:, 0, :].numpy()
        todos_embeddings.append(cls_batch)

        procesados = min(inicio + batch_size, len(textos))
        if procesados % (batch_size * 5) == 0 or procesados == len(textos):
            print(f"  Embeddings generados: {procesados}/{len(textos)}")

    return np.vstack(todos_embeddings)


def embedding_token(
    texto: str,
    palabra_objetivo: str,
    tokenizer,
    model,
    max_length: int = 256,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Retorna el embedding contextual de una palabra específica dentro de un texto.
    Útil para demostrar polisemia: la misma palabra obtiene vectores distintos
    según el contexto que la rodea.

    Returns
    -------
    (vector np.ndarray de 768d, lista de tokens de la oración)
    """
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]   # (seq_len, 768)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    objetivo_lower = palabra_objetivo.lower()

    # Buscar posición del token que corresponde a la palabra
    for i, tok in enumerate(tokens):
        tok_limpio = tok.replace("##", "").lower()
        if tok_limpio == objetivo_lower or objetivo_lower in tok_limpio:
            return hidden[i].numpy(), tokens

    # Fallback: coincidencia parcial
    for i, tok in enumerate(tokens):
        if objetivo_lower[:4] in tok.replace("##", "").lower():
            return hidden[i].numpy(), tokens

    return None, tokens


# ─── Análisis de polisemia ────────────────────────────────────────────────────

def analizar_polisemia(
    pares: List[Tuple[str, str]],
    tokenizer,
    model,
) -> pd.DataFrame:
    """
    Recibe una lista de (oración, palabra_objetivo) y demuestra cómo
    BETO genera vectores distintos para la misma palabra en distintos contextos.

    Parameters
    ----------
    pares : [(texto, palabra), ...]
            Agrupar varios pares de la misma palabra para comparar.

    Returns
    -------
    pd.DataFrame con columnas: oracion, palabra, similitud_con_anterior
    """
    resultados = []
    emb_anterior = None
    palabra_anterior = None

    for texto, palabra in pares:
        emb, tokens = embedding_token(texto, palabra, tokenizer, model)
        if emb is None:
            print(f"  No se encontró '{palabra}' en: {texto[:60]}")
            continue

        fila = {"oracion": texto[:70], "palabra": palabra, "tokens": tokens}

        if emb_anterior is not None and palabra == palabra_anterior:
            sim = cosine_similarity([emb_anterior], [emb])[0][0]
            fila["similitud_con_anterior"] = round(float(sim), 4)
            print(f"  Similitud '{palabra}' (ctx anterior vs actual): {sim:.4f}")
        else:
            fila["similitud_con_anterior"] = None

        resultados.append(fila)
        emb_anterior = emb
        palabra_anterior = palabra

    return pd.DataFrame(resultados)


# ─── Búsqueda semántica ───────────────────────────────────────────────────────

class BuscadorSemantico:
    """
    Sistema de búsqueda semántica: dado un texto de consulta,
    encuentra las canciones más similares usando embeddings BETO.

    Uso
    ---
        buscador = BuscadorSemantico(df, tokenizer, model)
        buscador.indexar()
        resultados = buscador.buscar("feel the rhythm in my soul", top_k=5)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        model,
        col_titulo: str  = "titulo",
        col_artista: str = "artista",
        col_genero: str  = "genero",
        col_letra: str   = "letra",
    ):
        self._df       = df.reset_index(drop=True)
        self._tokenizer = tokenizer
        self._model    = model
        self._col_titulo   = col_titulo
        self._col_artista  = col_artista
        self._col_genero   = col_genero
        self._col_letra    = col_letra
        self._index: Optional[np.ndarray] = None

    def indexar(self, max_canciones: Optional[int] = None) -> None:
        """Genera y almacena los embeddings de todas las canciones del DataFrame."""
        df_idx = self._df if max_canciones is None else self._df.head(max_canciones)
        textos = df_idx[self._col_letra].fillna("").tolist()
        print(f"Indexando {len(textos)} canciones con BETO...")
        self._index = embedding_cls(textos, self._tokenizer, self._model)
        print(f"Índice listo: {self._index.shape}")

    def buscar(
        self, consulta: str, top_k: int = 5
    ) -> pd.DataFrame:
        """
        Busca las canciones más similares a la consulta.

        Returns
        -------
        pd.DataFrame con columnas: titulo, artista, genero, similitud
        """
        if self._index is None:
            raise RuntimeError("Llama indexar() antes de buscar().")

        emb_consulta = embedding_cls(
            [consulta], self._tokenizer, self._model
        )
        sims = cosine_similarity(emb_consulta, self._index)[0]

        indices_top = sims.argsort()[::-1][:top_k]
        filas = []
        for idx in indices_top:
            row = self._df.iloc[idx]
            filas.append({
                "titulo":    row[self._col_titulo],
                "artista":   row[self._col_artista],
                "genero":    row[self._col_genero],
                "similitud": round(float(sims[idx]), 4),
            })

        resultado = pd.DataFrame(filas)
        print(f"\nResultados para: '{consulta}'")
        print(resultado.to_string(index=False))
        return resultado


# ─── Masked Language Model ────────────────────────────────────────────────────

class AnalizadorMLM:
    """
    Usa la capacidad Masked Language Model de BETO para analizar
    cómo el modelo "completa" frases típicas de cada género musical.

    Uso
    ---
        mlm = AnalizadorMLM()
        mlm.predecir("Siento tu [MASK] en mi corazón", top_k=5)
        mlm.analizar_por_genero(plantillas, df, col_genero="genero", col_letra="letra")
    """

    def __init__(self, modelo: str = BETO_MODEL):
        print(f"Cargando pipeline fill-mask con BETO...")
        self._pipe = pipeline("fill-mask", model=modelo)
        print("Pipeline listo.\n")

    def predecir(
        self, oracion_con_mask: str, top_k: int = 5
    ) -> List[Dict]:
        """
        Predice las palabras más probables para [MASK] en la oración.

        Parameters
        ----------
        oracion_con_mask : oración con exactamente un token [MASK]
        top_k            : número de predicciones a retornar
        """
        resultados = self._pipe(oracion_con_mask, top_k=top_k)
        print(f"\nOración: {oracion_con_mask}")
        print(f"Predicciones BETO:")
        for r in resultados:
            print(f"  {r['token_str']:<20} prob: {r['score']:.4f}")
        return resultados

    def analizar_por_genero(
        self,
        plantillas: List[str],
        df: pd.DataFrame,
        col_genero: str = "genero",
        col_letra:  str = "letra",
        top_k: int = 5,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Construye plantillas contextuales usando palabras frecuentes de cada género
        y analiza las predicciones de BETO para cada plantilla.

        Parameters
        ----------
        plantillas : lista de frases con [MASK], ej:
                     ["I [MASK] you every night",
                      "feel the [MASK] in my heart"]
        df         : DataFrame con las canciones
        col_genero : columna del género
        col_letra  : columna de la letra

        Returns
        -------
        Dict { género → { plantilla → [predicciones BETO] } }
        """
        generos = df[col_genero].dropna().unique()
        resultado_total: Dict[str, Dict[str, List[Dict]]] = {}

        for genero in generos:
            print(f"\n{'='*50}")
            print(f"Género: {genero}")
            print('='*50)
            resultado_total[genero] = {}

            for plantilla in plantillas:
                print(f"\n  Plantilla: '{plantilla}'")
                try:
                    preds = self._pipe(plantilla, top_k=top_k)
                    resultado_total[genero][plantilla] = preds
                    for p in preds:
                        print(f"    {p['token_str']:<20} {p['score']:.4f}")
                except Exception as e:
                    print(f"    Error: {e}")
                    resultado_total[genero][plantilla] = []

        return resultado_total

    def palabras_frecuentes_por_genero(
        self,
        df: pd.DataFrame,
        col_genero: str = "genero",
        col_letra:  str = "letra",
        top_n: int = 20,
        stopwords_extra: Optional[List[str]] = None,
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Extrae las palabras más frecuentes de cada género (excluye stopwords
        comunes). Útil para construir plantillas de MLM contextualizadas.

        Returns
        -------
        Dict { género → [(palabra, frecuencia), ...] }
        """
        import re
        from collections import Counter

        stops = {
            "the","a","an","in","on","at","to","of","and","or","but","is","are",
            "was","were","be","been","being","have","has","had","do","does","did",
            "will","would","shall","should","may","might","must","can","could",
            "i","you","he","she","it","we","they","me","him","her","us","them",
            "my","your","his","its","our","their","this","that","these","those",
            "not","no","so","as","if","for","with","from","by","about","like",
            "just","get","got","go","know","think","want","make","see","come",
            "el","la","los","las","un","una","de","del","en","con","por","para",
            "que","y","o","pero","si","no","se","le","lo","su","mi","tu",
        }
        if stopwords_extra:
            stops.update(stopwords_extra)

        frecuencias: Dict[str, List[Tuple[str, int]]] = {}

        for genero in df[col_genero].dropna().unique():
            letras = df[df[col_genero] == genero][col_letra].dropna()
            tokens = []
            for letra in letras:
                palabras = re.sub(r"[^\w\s]", " ", str(letra).lower()).split()
                tokens.extend([p for p in palabras if p not in stops and len(p) > 2])

            conteo = Counter(tokens).most_common(top_n)
            frecuencias[genero] = conteo
            print(f"\nTop palabras en '{genero}': {[w for w,_ in conteo[:10]]}")

        return frecuencias


# ─── Actualización en MongoDB ─────────────────────────────────────────────────

def actualizar_beto_cls_mongodb(
    df: pd.DataFrame,
    tokenizer,
    model,
    col_id:    str = "id",
    col_letra: str = "letra",
    batch_size: int = 50,
) -> None:
    """
    Calcula embeddings BETO [CLS] para cada canción del DataFrame
    y actualiza el campo embeddings.beto_cls en MongoDB.
    """
    from src.database.MongoConnection import MongoConnection, MongoConfig
    from bson import ObjectId

    MongoConnection.connect()
    collection = MongoConnection.get_db()[MongoConfig.COLLECTION_CANCIONES]

    textos  = df[col_letra].fillna("").tolist()
    ids     = df[col_id].tolist()

    print(f"Generando embeddings BETO para {len(textos)} canciones...")
    embeddings = embedding_cls(textos, tokenizer, model, batch_size=batch_size)

    print("Actualizando MongoDB...")
    actualizados = 0
    for cancion_id, vector in zip(ids, embeddings):
        collection.update_one(
            {"_id": ObjectId(cancion_id)},
            {"$set": {"embeddings.beto_cls": vector.tolist()}},
        )
        actualizados += 1
        if actualizados % 100 == 0:
            print(f"  Actualizados: {actualizados}/{len(ids)}")

    print(f"\nTotal actualizados en MongoDB: {actualizados}")
    MongoConnection.disconnect()
