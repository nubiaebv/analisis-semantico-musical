from typing import Optional, List
from src.repositories.CancionRepository import CancionRepository
from src.entities.CancionEntity import CancionEntity, MetricasEntity, PosTagsEntity, EmbeddingsEntity


class CancionService:
    def __init__(self):
        self.repository = CancionRepository()

    def crear_cancion(
        self,
        titulo: str,
        artista: str,
        genero: str,
        anio: int,
        letra: str,
        idioma: str,
        fuente: str,
        url_fuente: str,
        metricas: MetricasEntity = None,
        pos_tags: PosTagsEntity = None,
        embeddings: EmbeddingsEntity = None
    ) -> CancionEntity:
        cancion = CancionEntity(
            titulo=titulo,
            artista=artista,
            genero=genero,
            anio=anio,
            letra=letra,
            idioma=idioma,
            fuente=fuente,
            url_fuente=url_fuente,
            metricas=metricas or MetricasEntity(),
            pos_tags=pos_tags or PosTagsEntity(),
            embeddings=embeddings or EmbeddingsEntity()
        )
        return self.repository.save(cancion)

    def obtener_cancion(self, cancion_id: str) -> Optional[CancionEntity]:
        cancion = self.repository.find_by_id(cancion_id)
        if not cancion:
            raise ValueError(f"Canción con id '{cancion_id}' no encontrada.")
        return cancion

    def obtener_por_artista(self, artista: str) -> List[CancionEntity]:
        return self.repository.find_by_artista(artista)

    def obtener_por_genero(self, genero: str) -> List[CancionEntity]:
        return self.repository.find_by_genero(genero)

    def obtener_por_idioma(self, idioma: str) -> List[CancionEntity]:
        return self.repository.find_by_idioma(idioma)

    def obtener_por_anio(self, anio: int) -> List[CancionEntity]:
        return self.repository.find_by_anio(anio)

    def obtener_todas(self) -> List[CancionEntity]:
        return self.repository.find_all()

    def actualizar_cancion(self, cancion: CancionEntity) -> bool:
        return self.repository.update(cancion)

    def eliminar_cancion(self, cancion_id: str) -> bool:
        return self.repository.delete(cancion_id)