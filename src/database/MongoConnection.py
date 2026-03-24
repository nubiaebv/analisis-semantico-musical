import os
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

class MongoConfig:
    URI: str = os.getenv("MONGO_URI")
    DB_NAME: str = os.getenv("MONGO_DB_NAME", "analisisMusical")
    COLLECTION_CANCIONES: str = "analisisMusical"

class MongoConnection:
    _client: MongoClient = None
    _db: Database = None

    @classmethod
    def connect(cls) -> Database:
        if cls._client is None:
            cls._client = MongoClient(
                MongoConfig.URI,
                server_api=ServerApi("1")
            )
            cls._client.admin.command("ping")
            cls._db = cls._client[MongoConfig.DB_NAME]
            print(f"✅ Conectado a MongoDB Atlas: {MongoConfig.DB_NAME}")
        return cls._db

    @classmethod
    def get_db(cls) -> Database:
        if cls._db is None:
            raise RuntimeError("DB no inicializada. Llama connect() primero.")
        return cls._db

    @classmethod
    def disconnect(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("🔌 Desconectado de MongoDB Atlas")