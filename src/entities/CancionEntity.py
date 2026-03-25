from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId


@dataclass
class PosTagsEntity:
    nltk: List = field(default_factory=list)
    spacy: List = field(default_factory=list)


@dataclass
class EmbeddingsEntity:
    word2vec_avg: List[float] = field(default_factory=list)
    beto_cls: List[float] = field(default_factory=list)


@dataclass
class MetricasEntity:
    num_palabras: int = 0
    densidad_lexica: float = 0.0
    ratio_sustantivos_verbos: float = 0.0


@dataclass
class CancionEntity:
    titulo: str
    artista: str
    genero: str                  # "Rock" | "Pop" | "Hip-Hop" | "Reggaetón" | "Balada"
    anio: int
    letra: str
    idioma: str                  # "es" | "en"
    fuente: str                  # "kaggle" | "scraping"
    url_fuente: str
    id: Optional[str]            = field(default=None)
    fecha_recopilacion: Optional[datetime] = field(default=None)
    pos_tags: PosTagsEntity      = field(default_factory=PosTagsEntity)
    embeddings: EmbeddingsEntity = field(default_factory=EmbeddingsEntity)
    metricas: MetricasEntity     = field(default_factory=MetricasEntity)

    @staticmethod
    def from_mongo(doc: dict) -> "CancionEntity":
        pos = doc.get("pos_tags", {})
        emb = doc.get("embeddings", {})
        met = doc.get("metricas", {})

        return CancionEntity(
            id=str(doc["_id"]),
            titulo=doc.get("titulo", ""),
            artista=doc.get("artista", ""),
            genero=doc.get("genero", ""),
            anio=doc.get("anio", 0),
            letra=doc.get("letra", ""),
            idioma=doc.get("idioma", ""),
            fuente=doc.get("fuente", ""),
            url_fuente=doc.get("url_fuente", ""),
            fecha_recopilacion=doc.get("fecha_recopilacion"),
            pos_tags=PosTagsEntity(
                nltk=pos.get("nltk", []),
                spacy=pos.get("spacy", [])
            ),
            embeddings=EmbeddingsEntity(
                word2vec_avg=emb.get("word2vec_avg", []),
                beto_cls=emb.get("beto_cls", [])
            ),
            metricas=MetricasEntity(
                num_palabras=met.get("num_palabras", 0),
                densidad_lexica=met.get("densidad_lexica", 0.0),
                ratio_sustantivos_verbos=met.get("ratio_sustantivos_verbos", 0.0)
            )
        )

    def to_mongo(self) -> dict:
        data = {
            "titulo": self.titulo,
            "artista": self.artista,
            "genero": self.genero,
            "anio": self.anio,
            "letra": self.letra,
            "idioma": self.idioma,
            "fuente": self.fuente,
            "url_fuente": self.url_fuente,
            "fecha_recopilacion": self.fecha_recopilacion or datetime.utcnow(),
            "pos_tags": {
                "nltk": self.pos_tags.nltk,
                "spacy": self.pos_tags.spacy
            },
            "embeddings": {
                "word2vec_avg": self.embeddings.word2vec_avg,
                "beto_cls": self.embeddings.beto_cls
            },
            "metricas": {
                "num_palabras": self.metricas.num_palabras,
                "densidad_lexica": self.metricas.densidad_lexica,
                "ratio_sustantivos_verbos": self.metricas.ratio_sustantivos_verbos
            }
        }
        if self.id:
            data["_id"] = ObjectId(self.id)
        return data