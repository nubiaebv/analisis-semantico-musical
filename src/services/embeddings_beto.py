"""
src/services/embeddings_beto.py
================================
Adaptador BetoService
----------------------
El dashboard (beto.py) espera una clase BetoService con la interfaz:

    svc = BetoService(model_name="bert-base-uncased", batch_size=32, max_length=128)
    svc.load_model()
    svc.polysemy_demo(word, contexts)  → [{"ctx_a", "ctx_b", "sim"}, ...]
    svc.fill_mask(sentence, top_k)     → [{"token", "score"}, ...]
    svc.semantic_search(query, emb, df, top_n, col_title, col_artist, col_genre)
        → [{"rank", "title", "artist", "genre", "score"}, ...]

Este módulo implementa esa interfaz usando el backend real:
    - CargadorBETO  (carga tokenizer + modelo)
    - embedding_cls / embedding_token  (genera embeddings)
    - AnalizadorMLM  (pipeline fill-mask)
    - BuscadorSemantico  (búsqueda semántica)

Los embeddings de documento se leen directamente de MongoDB a través de
consultar_base_datos, evitando archivos .npy externos.
"""

import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BetoService:
    """
    Wrapper de alto nivel para el modelo BERT del proyecto.

    Expone la interfaz que el dashboard necesita delegando al backend real
    (CargadorBETO, embedding_cls, embedding_token, AnalizadorMLM).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 16,
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self._tokenizer = None
        self._model     = None
        self._mlm       = None   # AnalizadorMLM

    # ── Carga ─────────────────────────────────────────────────────────────────

    def load_model(self) -> "BetoService":
        """Carga el tokenizer y el modelo BERT en memoria."""
        try:
            from src.embeddings.embeddings_beto import CargadorBETO
            cargador = CargadorBETO(modelo=self.model_name)
            self._tokenizer = cargador.tokenizer
            self._model     = cargador.model
            logger.info("BetoService: modelo %s cargado correctamente.", self.model_name)
        except Exception as e:
            logger.error("BetoService: no se pudo cargar el modelo: %s", e)
        return self

    def _lazy_load(self) -> bool:
        """Carga el modelo si todavía no está en memoria. Retorna True si listo."""
        if self._tokenizer is None or self._model is None:
            self.load_model()
        return self._tokenizer is not None and self._model is not None

    def _get_mlm(self):
        """Retorna el pipeline AnalizadorMLM (lo crea la primera vez)."""
        if self._mlm is None:
            try:
                from src.embeddings.embeddings_beto import AnalizadorMLM
                self._mlm = AnalizadorMLM(modelo=self.model_name)
            except Exception as e:
                logger.error("BetoService: no se pudo crear AnalizadorMLM: %s", e)
        return self._mlm

    # ── API pública ───────────────────────────────────────────────────────────

    def polysemy_demo(
        self,
        word: str,
        contexts: List[str],
    ) -> List[Dict]:
        """
        Calcula la similitud coseno entre todos los pares de contextos para
        la misma palabra, demostrando polisemia contextual.

        Parameters
        ----------
        word     : palabra objetivo (ej. "rock")
        contexts : lista de oraciones que contienen la palabra

        Returns
        -------
        Lista de dicts con claves: ctx_a, ctx_b, sim
        """
        if not self._lazy_load():
            return []

        from src.embeddings.embeddings_beto import embedding_token
        from sklearn.metrics.pairwise import cosine_similarity

        # Calcular embedding contextual para cada oración
        embeddings = []
        valid_ctx  = []
        for ctx in contexts:
            if not ctx or not ctx.strip():
                continue
            emb, _ = embedding_token(
                ctx, word, self._tokenizer, self._model,
                max_length=self.max_length,
            )
            if emb is not None:
                embeddings.append(emb)
                valid_ctx.append(ctx)

        if len(embeddings) < 2:
            return []

        # Comparar todos los pares
        results = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(
                    cosine_similarity(
                        [embeddings[i]], [embeddings[j]]
                    )[0][0]
                )
                results.append({
                    "ctx_a": valid_ctx[i],
                    "ctx_b": valid_ctx[j],
                    "sim":   round(sim, 4),
                })
        return results

    def fill_mask(
        self,
        sentence: str,
        top_k: int = 6,
    ) -> List[Dict]:
        """
        Predice la palabra enmascarada usando el pipeline fill-mask de BERT.

        Parameters
        ----------
        sentence : oración con exactamente un [MASK]
        top_k    : número de predicciones

        Returns
        -------
        Lista de dicts con claves: token, score
        """
        mlm = self._get_mlm()
        if mlm is None:
            return []
        try:
            preds = mlm.predecir(sentence, top_k=top_k)
            # AnalizadorMLM retorna dicts con "token_str" y "score"
            # El dashboard espera "token" y "score"
            return [
                {"token": p.get("token_str", p.get("token", "")), "score": p["score"]}
                for p in preds
                if "score" in p
            ]
        except Exception as e:
            logger.error("fill_mask error: %s", e)
            return []

    def semantic_search(
        self,
        query: str,
        embeddings: np.ndarray,
        df,
        top_n: int = 5,
        col_title:  str = "titulo",
        col_artist: str = "artista",
        col_genre:  str = "genero",
    ) -> List[Dict]:
        """
        Busca las canciones más similares a `query` usando los embeddings
        BERT pre-calculados que ya están en MongoDB.

        Parameters
        ----------
        query      : texto libre de la consulta
        embeddings : np.ndarray (n_canciones, 768) — leído de MongoDB
        df         : DataFrame de canciones (mismo orden que embeddings)
        top_n      : número de resultados
        col_*      : nombres de columnas en df

        Returns
        -------
        Lista de dicts con claves: rank, title, artist, genre, score
        """
        if not self._lazy_load():
            return []
        if embeddings is None or len(embeddings) == 0:
            return []

        try:
            from src.embeddings.embeddings_beto import embedding_cls
            from sklearn.metrics.pairwise import cosine_similarity

            # Embedding de la consulta
            q_emb = embedding_cls(
                [query], self._tokenizer, self._model,
                max_length=self.max_length, batch_size=1,
            )
            sims = cosine_similarity(q_emb, embeddings)[0]
            top_idx = sims.argsort()[::-1][:top_n]

            results = []
            for rank, idx in enumerate(top_idx, 1):
                row = df.iloc[int(idx)]
                results.append({
                    "rank":   rank,
                    "title":  str(row.get(col_title,  "—")),
                    "artist": str(row.get(col_artist, "—")),
                    "genre":  str(row.get(col_genre,  "—")),
                    "score":  round(float(sims[idx]), 4),
                })
            return results

        except Exception as e:
            logger.error("semantic_search error: %s", e)
            return []
