from typing import Optional, List
from bson import ObjectId
from src.database.MongoConnection import MongoConnection, MongoConfig
from src.entities.CancionEntity import CancionEntity


class CancionRepository:
    def __init__(self):
        self.collection = MongoConnection.get_db()[MongoConfig.COLLECTION_CANCIONES]

    def find_by_id(self, cancion_id: str) -> Optional[CancionEntity]:
        doc = self.collection.find_one({"_id": ObjectId(cancion_id)})
        return CancionEntity.from_mongo(doc) if doc else None

    def find_by_artista(self, artista: str) -> List[CancionEntity]:
        docs = self.collection.find({"artista": artista})
        return [CancionEntity.from_mongo(d) for d in docs]

    def find_by_genero(self, genero: str) -> List[CancionEntity]:
        docs = self.collection.find({"genero": genero})
        return [CancionEntity.from_mongo(d) for d in docs]

    def find_by_idioma(self, idioma: str) -> List[CancionEntity]:
        docs = self.collection.find({"idioma": idioma})
        return [CancionEntity.from_mongo(d) for d in docs]

    def find_by_anio(self, anio: int) -> List[CancionEntity]:
        docs = self.collection.find({"anio": anio})
        return [CancionEntity.from_mongo(d) for d in docs]

    def find_by_fuente(self, fuente: str) -> List[CancionEntity]:
        docs = self.collection.find({"fuente": fuente})
        return [CancionEntity.from_mongo(d) for d in docs]

    def find_all(self) -> List[CancionEntity]:
        return [CancionEntity.from_mongo(d) for d in self.collection.find()]

    def find_by_generos(self, generos: List[str]) -> List[CancionEntity]:
        docs = self.collection.find({"genero": {"$in": generos}})
        return [CancionEntity.from_mongo(d) for d in docs]

    def save(self, cancion: CancionEntity) -> CancionEntity:
        result = self.collection.insert_one(cancion.to_mongo())
        cancion.id = str(result.inserted_id)
        return cancion

    def update(self, cancion: CancionEntity) -> bool:
        result = self.collection.update_one(
            {"_id": ObjectId(cancion.id)},
            {"$set": cancion.to_mongo()}
        )
        return result.modified_count > 0

    def delete(self, cancion_id: str) -> bool:
        result = self.collection.delete_one({"_id": ObjectId(cancion_id)})
        return result.deleted_count > 0