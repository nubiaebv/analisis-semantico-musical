import pandas as pd
import numpy as np
from typing import Optional, List

from src.database.MongoConnection import MongoConnection
from src.services.CancionService import CancionService
from src.entities.CancionEntity import CancionEntity

class consultar_base_datos:
    """
    Carga canciones desde MongoDB (vía CancionService) y las expone
    como un DataFrame de pandas con columnas planas.
    """

    def __init__(self):
        MongoConnection.connect()
        self._service = CancionService()
        self._df: Optional[pd.DataFrame] = None


    # ─── Propiedades ─────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("No hay datos cargados. Llama primero a cargar_*().")
        return self._df

    @property
    def shape(self):
        return self.df.shape

    @property
    def columns(self):
        return self.df.columns.tolist()

    # ─── Métodos de carga ────────────────────────────────────────────────────

    def cargar_todas(self) -> "CancionDataFrame":
        canciones = self._service.obtener_todas()
        self._df = self._build(canciones)
        return self

    def cargar_por_artista(self, artista: str) -> "consultar_base_datos":
        canciones = self._service.obtener_por_artista(artista)
        self._df = self._build(canciones)
        return self

    def cargar_por_genero(self, genero: str) -> "consultar_base_datos":
        canciones = self._service.obtener_por_genero(genero)
        self._df = self._build(canciones)
        return self

    def cargar_por_idioma(self, idioma: str) -> "consultar_base_datos":
        canciones = self._service.obtener_por_idioma(idioma)
        self._df = self._build(canciones)
        return self

    def cargar_por_anio(self, anio: int) -> "consultar_base_datos":
        canciones = self._service.obtener_por_anio(anio)
        self._df = self._build(canciones)
        return self

    def cargar_por_id(self, cancion_id: str) -> "consultar_base_datos":
        cancion = self._service.obtener_cancion(cancion_id)
        self._df = self._build([cancion])
        return self
    def cargar_por_generos(self, generos: List[str]) -> "consultar_base_datos":
        canciones = self._service.obtener_por_generos(generos)
        self._df = self._build(canciones)
        return self

    # ─── Accesos rápidos ─────────────────────────────────────────────────────

    def metricas(self):
        """Solo las columnas de métricas numéricas."""
        cols = ["id", "titulo", "artista",
                "metricas_num_palabras",
                "metricas_densidad_lexica",
                "metricas_ratio_sust_verbos"]
        return self.df[cols]

    def embeddings_word2vec(self):
        """Matriz (n_canciones, dim) con los vectores word2vec."""
        return np.array(self.df["embeddings_word2vec_avg"].tolist())

    def embeddings_beto(self):
        """Matriz (n_canciones, dim) con los vectores BETO CLS."""
        return np.array(self.df["embeddings_beto_cls"].tolist())

    def sin_vectores(self):
        """DataFrame sin columnas de embeddings ni pos_tags (apto para CSV)."""
        cols_excluir = [
            "pos_tags_nltk", "pos_tags_spacy",
            "embeddings_word2vec_avg", "embeddings_beto_cls"
        ]
        return self.df.drop(columns=cols_excluir)

    def resumen(self):
        print(f"Canciones cargadas : {len(self.df)}")
        print(f"Columnas           : {len(self.columns)}")
        print(f"\n── Vista previa ──")
        print(self.sin_vectores().head(5).to_string(index=False))
        print(f"\n── Estadísticas de métricas ──")
        print(self.metricas().describe())

    # ─── Interno ─────────────────────────────────────────────────────────────

    @staticmethod
    def _cancion_to_dict(c: CancionEntity) -> dict:
        return {
            "id": c.id,
            "titulo": c.titulo,
            "artista": c.artista,
            "genero": c.genero,
            "anio": c.anio,
            "letra": c.letra,
            "idioma": c.idioma,
            "fuente": c.fuente,
            "url_fuente": c.url_fuente,
            "fecha_recopilacion": c.fecha_recopilacion,
            "metricas_num_palabras": c.metricas.num_palabras,
            "metricas_densidad_lexica": c.metricas.densidad_lexica,
            "metricas_ratio_sust_verbos": c.metricas.ratio_sustantivos_verbos,
            "pos_tags_nltk": c.pos_tags.nltk,
            "pos_tags_spacy": c.pos_tags.spacy,
            "embeddings_word2vec_avg": c.embeddings.word2vec_avg,
            "embeddings_beto_cls": c.embeddings.beto_cls,
        }

    @staticmethod
    def _build(canciones: List[CancionEntity]) -> pd.DataFrame:
        if not canciones:
            return pd.DataFrame()

        df = pd.DataFrame([consultar_base_datos._cancion_to_dict(c) for c in canciones])
        df["anio"] = df["anio"].astype("Int64")
        df["metricas_num_palabras"] = df["metricas_num_palabras"].astype("Int64")
        df["metricas_densidad_lexica"] = df["metricas_densidad_lexica"].astype(float)
        df["metricas_ratio_sust_verbos"] = df["metricas_ratio_sust_verbos"].astype(float)
        df["fecha_recopilacion"] = pd.to_datetime(df["fecha_recopilacion"], utc=True)
        return df
