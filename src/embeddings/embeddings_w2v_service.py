"""
src/embeddings/embeddings_w2v_service.py
========================================
Adaptador Word2VecService
--------------------------
El dashboard (word2vec.py) espera una clase Word2VecService con la interfaz:

    svc = Word2VecService()
    svc.load(models_dir, prefix="w2v")
    svc.vocab_size          → int
    svc.vector_size         → int
    svc.cbow                → modelo cargado
    svc.skipgram            → modelo cargado
    svc.most_similar(word, topn, model)   → [(word, score), ...]
    svc.analogy(a, b, c, topn)            → [(word, score), ...]
    svc.genre_similarity_matrix(df, ...)  → pd.DataFrame

Este módulo implementa esa interfaz usando el backend real
(EntrenadorWord2Vec / AnalizadorWord2Vec de embeddings_w2v.py).

Los embeddings de documento se leen directamente de MongoDB a través de
consultar_base_datos, evitando archivos .npy externos.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)


class Word2VecService:
    """
    Wrapper de alto nivel para los modelos Word2Vec del proyecto.

    Carga los modelos CBOW y Skip-Gram desde data/results/ (generados
    por el notebook de entrenamiento) y expone la interfaz que el
    dashboard necesita.

    Si los modelos no están en disco todavía, los entrena on-the-fly
    usando el corpus de MongoDB, los guarda y los cachea en memoria.
    """

    def __init__(self):
        self.cbow     = None   # gensim Word2Vec
        self.skipgram = None   # gensim Word2Vec
        self._ana_cbow: Optional[object] = None   # AnalizadorWord2Vec
        self._ana_sg:   Optional[object] = None   # AnalizadorWord2Vec

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        modelo = self.skipgram or self.cbow
        if modelo is None:
            return 0
        return len(modelo.wv)

    @property
    def vector_size(self) -> int:
        modelo = self.skipgram or self.cbow
        if modelo is None:
            return 0
        return modelo.wv.vector_size

    # ── Carga ─────────────────────────────────────────────────────────────────

    def load(self, models_dir, prefix: str = "w2v") -> "Word2VecService":
        """
        Carga los modelos desde disco (rutas: <models_dir>/<prefix>_cbow.model
        y <models_dir>/<prefix>_skipgram.model).

        Si no existen en disco, intenta entrenarlos on-the-fly usando el corpus
        de MongoDB y los guarda para usos futuros.
        """
        from gensim.models import Word2Vec as _W2V
        from src.embeddings.embeddings_w2v import AnalizadorWord2Vec, EntrenadorWord2Vec

        models_dir = Path(models_dir)
        ruta_cbow  = models_dir / f"{prefix}_cbow.model"
        ruta_sg    = models_dir / f"{prefix}_skipgram.model"

        # ── Intentar cargar desde disco ──────────────────────────────────────
        if ruta_cbow.exists():
            try:
                self.cbow = _W2V.load(str(ruta_cbow))
                logger.info("CBOW cargado: %d palabras", len(self.cbow.wv))
            except Exception as e:
                logger.warning("Error cargando CBOW: %s", e)

        if ruta_sg.exists():
            try:
                self.skipgram = _W2V.load(str(ruta_sg))
                logger.info("Skip-Gram cargado: %d palabras", len(self.skipgram.wv))
            except Exception as e:
                logger.warning("Error cargando Skip-Gram: %s", e)

        # ── Fallback: entrenar on-the-fly si no hay modelos en disco ─────────
        if self.cbow is None and self.skipgram is None:
            logger.warning(
                "Modelos no encontrados en %s — entrenando on-the-fly desde MongoDB…",
                models_dir,
            )
            try:
                self._entrenar_y_guardar(models_dir, prefix)
            except Exception as e:
                logger.error("No se pudo entrenar on-the-fly: %s", e)
                return self

        # ── Crear analizadores ────────────────────────────────────────────────
        if self.cbow:
            self._ana_cbow = AnalizadorWord2Vec(self.cbow)
        if self.skipgram:
            self._ana_sg = AnalizadorWord2Vec(self.skipgram)

        return self

    def _entrenar_y_guardar(self, models_dir: Path, prefix: str) -> None:
        """Entrena CBOW + Skip-Gram con el corpus de MongoDB y los persiste."""
        from src.entities.consultar_base_datos import consultar_base_datos
        from src.embeddings.embeddings_w2v import EntrenadorWord2Vec

        logger.info("Cargando corpus desde MongoDB para entrenamiento…")
        cdb = consultar_base_datos().cargar_por_generos(
            ["pop", "alternative pop", "hip hop", "alternative rock", "dance pop", "rock"]
        )
        df = cdb.df

        entrenador = EntrenadorWord2Vec(df, col_letra="letra")
        entrenador.entrenar(vector_size=150, window=5, min_count=2, epochs=10)

        models_dir.mkdir(parents=True, exist_ok=True)
        entrenador.guardar(str(models_dir))

        # Renombrar con el prefix correcto si es distinto de "w2v"
        if prefix != "w2v":
            for sufijo in ("cbow", "skipgram"):
                src = models_dir / f"w2v_{sufijo}.model"
                dst = models_dir / f"{prefix}_{sufijo}.model"
                if src.exists() and not dst.exists():
                    src.rename(dst)

        self.cbow     = entrenador.modelo_cbow
        self.skipgram = entrenador.modelo_sg
        logger.info("Modelos entrenados y guardados en %s", models_dir)

    # ── API pública ───────────────────────────────────────────────────────────

    def _get_analizador(self, model_key: str):
        """Retorna el analizador correcto según 'skipgram' o 'cbow'."""
        if model_key == "skipgram" and self._ana_sg:
            return self._ana_sg
        if model_key == "cbow" and self._ana_cbow:
            return self._ana_cbow
        # Fallback al que esté disponible
        return self._ana_sg or self._ana_cbow

    def most_similar(
        self,
        word: str,
        topn: int = 12,
        model: str = "skipgram",
    ) -> List[Tuple[str, float]]:
        """
        Retorna las `topn` palabras más cercanas a `word`.

        Parameters
        ----------
        word  : palabra a consultar (minúsculas)
        topn  : número de vecinos
        model : "skipgram" | "cbow"
        """
        ana = self._get_analizador(model)
        if ana is None:
            return []
        try:
            return ana.palabras_similares(word, topn=topn)
        except Exception as e:
            logger.debug("most_similar error: %s", e)
            return []

    def analogy(
        self,
        a: str,
        b: str,
        c: str,
        topn: int = 5,
        model: str = "skipgram",
    ) -> List[Tuple[str, float]]:
        """
        Resuelve la analogía vectorial: b − a + c ≈ ???
        (el orden sigue la convención del dashboard: A−B+C)

        Parameters
        ----------
        a, b, c : palabras de la analogía
        topn    : número de resultados
        model   : "skipgram" | "cbow"
        """
        ana = self._get_analizador(model)
        if ana is None:
            return []
        try:
            # AnalizadorWord2Vec.analogia(positivo1, positivo2, negativo)
            # La fórmula del dashboard es: b − a + c
            # → positivo1=c, positivo2=b, negativo=a
            return ana.analogia(c, b, a, topn=topn)
        except Exception as e:
            logger.debug("analogy error: %s", e)
            return []

    def genre_similarity_matrix(
        self,
        df: pd.DataFrame,
        col_lyrics: str = "letra",
        col_genre:  str = "genero",
    ) -> pd.DataFrame:
        """
        Calcula la matriz de similitud coseno entre géneros usando el
        vector promedio de cada género con el modelo Skip-Gram (o CBOW).

        Returns
        -------
        pd.DataFrame  (géneros × géneros), valores en [0, 1]
        """
        ana = self._get_analizador("skipgram")
        if ana is None:
            return pd.DataFrame()
        try:
            return ana.similitud_entre_generos(df, col_letra=col_lyrics, col_genero=col_genre)
        except Exception as e:
            logger.error("genre_similarity_matrix error: %s", e)
            return pd.DataFrame()
