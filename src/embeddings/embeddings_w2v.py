
import re
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from pathlib import Path

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ─── Utilidades de texto ──────────────────────────────────────────────────────

def _tokenizar(texto: str) -> List[str]:
    """Tokeniza un texto eliminando puntuación y pasando a minúsculas."""
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    return [t for t in texto.split() if len(t) > 1]


def _preparar_corpus(df: pd.DataFrame, col_letra: str = "letra") -> List[List[str]]:
    """
    Convierte el DataFrame en una lista de listas de tokens
    lista para entrenar Word2Vec.
    """
    corpus = []
    for texto in df[col_letra].dropna():
        tokens = _tokenizar(str(texto))
        if tokens:
            corpus.append(tokens)
    print(f"Corpus preparado: {len(corpus):,} canciones | "
          f"{sum(len(s) for s in corpus):,} tokens totales")
    return corpus


# ─── Entrenamiento ────────────────────────────────────────────────────────────

class EntrenadorWord2Vec:
    """
    Entrena y gestiona modelos Word2Vec (CBOW y Skip-Gram)
    sobre el corpus musical.

    """

    def __init__(self, df: pd.DataFrame, col_letra: str = "letra"):
        self._df = df
        self._corpus = _preparar_corpus(df, col_letra)
        self.modelo_cbow: Optional[Word2Vec] = None
        self.modelo_sg: Optional[Word2Vec] = None

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def entrenar_cbow(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ) -> Word2Vec:
        """Entrena un modelo CBOW (sg=0)."""
        print("\nEntrenando CBOW...")
        self.modelo_cbow = Word2Vec(
            sentences=self._corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=0,          # CBOW
            epochs=epochs,
        )
        print(f"  Vocabulario: {len(self.modelo_cbow.wv):,} palabras")
        print(f"  Dimensión:   {vector_size}d")
        return self.modelo_cbow

    def entrenar_skip_gram(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ) -> Word2Vec:
        """Entrena un modelo Skip-Gram (sg=1)."""
        print("\nEntrenando Skip-Gram...")
        self.modelo_sg = Word2Vec(
            sentences=self._corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,          # Skip-Gram
            epochs=epochs,
        )
        print(f"  Vocabulario: {len(self.modelo_sg.wv):,} palabras")
        print(f"  Dimensión:   {vector_size}d")
        return self.modelo_sg

    def entrenar(self, **kwargs) -> None:
        """Entrena ambos modelos (CBOW y Skip-Gram)."""
        self.entrenar_cbow(**kwargs)
        self.entrenar_skip_gram(**kwargs)
        print("\nAmbos modelos entrenados.")

    # ── Persistencia ──────────────────────────────────────────────────────────

    def guardar(self, directorio: str = "data/results") -> None:
        """Guarda los modelos entrenados en disco."""
        Path(directorio).mkdir(parents=True, exist_ok=True)
        if self.modelo_cbow:
            ruta = f"{directorio}/w2v_cbow.model"
            self.modelo_cbow.save(ruta)
            print(f"CBOW guardado en: {ruta}")
        if self.modelo_sg:
            ruta = f"{directorio}/w2v_skipgram.model"
            self.modelo_sg.save(ruta)
            print(f"Skip-Gram guardado en: {ruta}")

    @staticmethod
    def cargar(directorio: str = "models/") -> Tuple[Optional[Word2Vec], Optional[Word2Vec]]:
        """Carga modelos guardados desde disco."""
        cbow, sg = None, None
        ruta_cbow = f"{directorio}/w2v_cbow.model"
        ruta_sg   = f"{directorio}/w2v_skipgram.model"
        if Path(ruta_cbow).exists():
            cbow = Word2Vec.load(ruta_cbow)
            print(f"CBOW cargado: {len(cbow.wv):,} palabras")
        if Path(ruta_sg).exists():
            sg = Word2Vec.load(ruta_sg)
            print(f"Skip-Gram cargado: {len(sg.wv):,} palabras")
        return cbow, sg


# ─── Análisis semántico ───────────────────────────────────────────────────────

class AnalizadorWord2Vec:
    """
    Herramientas de análisis semántico usando un modelo Word2Vec entrenado.

    """

    def __init__(self, modelo: Word2Vec):
        self.modelo = modelo
        self.wv = modelo.wv

    # ── Similitud ─────────────────────────────────────────────────────────────

    def palabras_similares(
        self, palabra: str, topn: int = 10
    ) -> List[Tuple[str, float]]:
        """Retorna las palabras más similares a una dada."""
        if palabra not in self.wv:
            print(f"'{palabra}' no está en el vocabulario.")
            return []
        resultados = self.wv.most_similar(palabra, topn=topn)
        print(f"\nPalabras más similares a '{palabra}':")
        for w, s in resultados:
            print(f"  {w:<20} {s:.4f}")
        return resultados

    # ── Analogías ─────────────────────────────────────────────────────────────

    def analogia(
        self, positivo1: str, positivo2: str, negativo: str, topn: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Resuelve: positivo2 - negativo + positivo1 ≈ ???
        Ejemplo:  king  - man + woman ≈ queen
        """
        try:
            resultado = self.wv.most_similar(
                positive=[positivo1, positivo2],
                negative=[negativo],
                topn=topn,
            )
            print(f"\nAnalogía: {positivo2} - {negativo} + {positivo1} ≈")
            for w, s in resultado:
                print(f"  {w:<20} {s:.4f}")
            return resultado
        except KeyError as e:
            print(f"Palabra no encontrada en vocabulario: {e}")
            return []

    # ── Campos semánticos por género ──────────────────────────────────────────

    def campos_semanticos_por_genero(
        self,
        df: pd.DataFrame,
        col_letra: str = "letra",
        col_genero: str = "genero",
        palabras_clave: Optional[List[str]] = None,
        topn: int = 8,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Entrena un modelo Word2Vec por género y retorna los vecinos
        semánticos de las palabras clave para cada género.
        """
        if palabras_clave is None:
            palabras_clave = ["love", "life", "night", "heart", "feel"]

        resultados: Dict[str, List[Tuple[str, float]]] = {}
        generos = df[col_genero].dropna().unique()

        for genero in generos:
            corpus_genero = _preparar_corpus(
                df[df[col_genero] == genero], col_letra
            )
            if len(corpus_genero) < 5:
                continue

            modelo_g = Word2Vec(
                sentences=corpus_genero,
                vector_size=100,
                window=5,
                min_count=2,
                sg=0,
                epochs=10,
                workers=4,
            )
            print(f"\n── {genero} ({len(corpus_genero)} canciones) ──")
            resultados[genero] = {}
            for palabra in palabras_clave:
                if palabra in modelo_g.wv:
                    vecinos = modelo_g.wv.most_similar(palabra, topn=topn)
                    resultados[genero][palabra] = vecinos
                    top5 = [w for w, _ in vecinos[:5]]
                    print(f"  {palabra:<12} → {top5}")

        return resultados

    # ── Similitud entre géneros ───────────────────────────────────────────────

    def similitud_entre_generos(
        self,
        df: pd.DataFrame,
        col_letra: str = "letra",
        col_genero: str = "genero",
    ) -> pd.DataFrame:
        """
        Calcula el vector promedio de cada género y la distancia coseno
        entre todos los pares de géneros.

        """
        vectores_genero: Dict[str, np.ndarray] = {}
        generos = df[col_genero].dropna().unique()

        for genero in generos:
            letras = df[df[col_genero] == genero][col_letra].dropna()
            vecs = []
            for letra in letras:
                tokens = _tokenizar(str(letra))
                for token in tokens:
                    if token in self.wv:
                        vecs.append(self.wv[token])
            if vecs:
                vectores_genero[genero] = np.mean(vecs, axis=0)

        generos_con_vec = list(vectores_genero.keys())
        matriz = np.array([vectores_genero[g] for g in generos_con_vec])
        sim_matrix = cosine_similarity(matriz)

        df_sim = pd.DataFrame(sim_matrix, index=generos_con_vec, columns=generos_con_vec)
        print("\nSimilitud coseno entre géneros:")
        print(df_sim.round(4).to_string())
        return df_sim

    # ── Vocabulario exclusivo por género ──────────────────────────────────────

    def vocabulario_exclusivo(
        self,
        df: pd.DataFrame,
        col_letra: str = "letra",
        col_genero: str = "genero",
        top_n: int = 20,
    ) -> Dict[str, List[str]]:
        """
        Identifica las palabras más características de cada género
        calculando cuáles están más cerca del centroide de ese género
        y más lejos del centroide global.
        """
        generos = df[col_genero].dropna().unique()

        # Centroide global
        all_vecs = []
        for letra in df[col_letra].dropna():
            for t in _tokenizar(str(letra)):
                if t in self.wv:
                    all_vecs.append(self.wv[t])
        centroide_global = np.mean(all_vecs, axis=0) if all_vecs else None

        exclusivo: Dict[str, List[str]] = {}

        for genero in generos:
            letras = df[df[col_genero] == genero][col_letra].dropna()
            tokens_genero = []
            for letra in letras:
                tokens_genero.extend(_tokenizar(str(letra)))

            vocab_genero = [t for t in set(tokens_genero) if t in self.wv]
            if not vocab_genero or centroide_global is None:
                continue

            # Centroide del género
            centroide_g = np.mean([self.wv[t] for t in vocab_genero], axis=0)

            # Puntuación: cercanía al centroide propio - cercanía al centroide global
            scores = {}
            for token in vocab_genero:
                v = self.wv[token].reshape(1, -1)
                cerca_g = cosine_similarity(v, centroide_g.reshape(1, -1))[0][0]
                cerca_global = cosine_similarity(v, centroide_global.reshape(1, -1))[0][0]
                scores[token] = cerca_g - cerca_global

            top = sorted(scores, key=scores.get, reverse=True)[:top_n]
            exclusivo[genero] = top
            print(f"\nVocabulario exclusivo de '{genero}':")
            print(f"  {top}")

        return exclusivo


# ─── Embeddings para MongoDB ──────────────────────────────────────────────────

def calcular_vector_promedio(
    texto: str, wv, stopwords: Optional[set] = None
) -> List[float]:
    """
    Calcula el vector promedio de un texto usando los vectores del modelo.
    Filtra stopwords si se proveen.
    """
    tokens = _tokenizar(texto)
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    vecs = [wv[t] for t in tokens if t in wv]
    if vecs:
        return np.mean(vecs, axis=0).tolist()
    return []


def actualizar_embeddings_mongodb(
    df: pd.DataFrame,
    modelo: Word2Vec,
    col_id: str = "id",
    col_letra: str = "letra",
    batch_size: int = 100,
) -> None:
    """
    Calcula el vector promedio word2vec para cada canción del DataFrame
    y actualiza el campo embeddings.word2vec_avg en MongoDB.
    """
    from src.database.MongoConnection import MongoConnection, MongoConfig
    from bson import ObjectId

    MongoConnection.connect()
    collection = MongoConnection.get_db()[MongoConfig.COLLECTION_CANCIONES]

    actualizados = 0
    for i, row in df.iterrows():
        vector = calcular_vector_promedio(str(row[col_letra]), modelo.wv)
        if not vector:
            continue

        collection.update_one(
            {"_id": ObjectId(row[col_id])},
            {"$set": {"embeddings.word2vec_avg": vector}},
        )
        actualizados += 1
        if actualizados % batch_size == 0:
            print(f"  Actualizados: {actualizados}/{len(df)}")

    print(f"\nTotal actualizados en MongoDB: {actualizados}")
    MongoConnection.disconnect()
