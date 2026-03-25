
import pandas as pd
from src.database.MongoConnection import MongoConnection
from src.entities.CancionEntity import (
    CancionEntity, MetricasEntity, PosTagsEntity, EmbeddingsEntity
)


class insertar_base_datos:
    """
    Clase para transformar un DataFrame de canciones
    y cargarlo en MongoDB Atlas por lotes.
    """

    def __init__(self, batch_size: int = 100, archivo=True):
        self.batch_size = batch_size
        self._archivo = archivo

    # ─────────────────────────────────────────
    # MÉTRICAS
    # ─────────────────────────────────────────
    def _calcular_ratio_sustantivos_verbos(self, pos_tags_spacy: list) -> float:
        """
        Calcula ratio sustantivos/verbos desde pos_tags de spaCy.
        Formato: [(token, pos), (token, pos), ...]
        NOUN, PROPN = sustantivos | VERB, AUX = verbos
        """
        sustantivos = sum(
            1 for item in pos_tags_spacy
            if isinstance(item, tuple) and len(item) >= 2
            and item[1] in ("NOUN", "PROPN")
        )
        verbos = sum(
            1 for item in pos_tags_spacy
            if isinstance(item, tuple) and len(item) >= 2
            and item[1] in ("VERB", "AUX")
        )
        return round(sustantivos / verbos, 4) if verbos > 0 else 0.0

    def _calcular_metricas(self, letra: str, pos_tags_spacy: list) -> MetricasEntity:
        """Calcula métricas a partir de la letra y los pos_tags de spaCy."""
        palabras = letra.split()
        num_palabras = len(palabras)
        palabras_unicas = set(palabras)
        densidad_lexica = len(palabras_unicas) / num_palabras if num_palabras > 0 else 0.0

        return MetricasEntity(
            num_palabras=num_palabras,
            densidad_lexica=round(densidad_lexica, 4),
            ratio_sustantivos_verbos=self._calcular_ratio_sustantivos_verbos(pos_tags_spacy)
        )

    # ─────────────────────────────────────────
    # TRANSFORMACIÓN
    # ─────────────────────────────────────────
    def _fila_a_entidad(self, row: pd.Series) -> CancionEntity:
        """Convierte una fila del DataFrame a una CancionEntity."""
        spacy_tags = row["Lematizado_Spacy"] if isinstance(row["Lematizado_Spacy"], list) else []
        nltk_tags  = row["Lematizado_nltk"]  if isinstance(row["Lematizado_nltk"],  list) else []
        metricas   = self._calcular_metricas(row["letra_cancion"], spacy_tags)
        if self._archivo:
            origen = "Kaggle"
            url_fuente = "https://www.kaggle.com/code/sajithdherath/starter-380-000-lyrics-from-24ddf566-9"
        else:
            origen = "WebScraping"
            url_fuente = "https://www.azlyrics.com"


        return CancionEntity(
            titulo=row["nombre_cancion"],
            artista=row["Artist"],
            genero=row["Genero"],
            anio=int(row["Periodo"]),
            letra=row["letra_cancion"],
            idioma=row["idioma"],
            fuente=origen,
            url_fuente=url_fuente,
            metricas=metricas,
            pos_tags=PosTagsEntity(spacy=spacy_tags, nltk=nltk_tags),
            embeddings=EmbeddingsEntity()
        )

    def _df_a_entidades(self, df: pd.DataFrame) -> list[CancionEntity]:
        """Transforma todas las filas del DataFrame a entidades."""
        entidades = []
        for _, row in df.iterrows():
            try:
                entidades.append(self._fila_a_entidad(row))
            except Exception as e:
                print(f"Error en fila '{row.get('nombre_cancion', '?')}': {e}")
        return entidades

    # ─────────────────────────────────────────
    # INSERCIÓN
    # ─────────────────────────────────────────
    def insertar(self, df: pd.DataFrame):
        """Punto de entrada principal: transforma e inserta el DataFrame."""
        print(f"\nIniciando carga de {len(df)} canciones...")

        MongoConnection.connect()
        collection = MongoConnection.get_db()["analisisMusical"]

        entidades  = self._df_a_entidades(df)
        documentos = [e.to_mongo() for e in entidades]

        total      = len(documentos)
        insertados = 0

        for i in range(0, total, self.batch_size):
            lote = documentos[i:i + self.batch_size]
            collection.insert_many(lote)
            insertados += len(lote)
            print(f"Insertados {insertados}/{total}")

        print(f"Carga completa: {total} canciones insertadas")
        MongoConnection.disconnect()


