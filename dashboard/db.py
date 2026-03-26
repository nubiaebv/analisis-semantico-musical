"""
dashboard/db.py
===============
Módulo de conexión a MongoDB Atlas para el Dashboard.
Reutiliza consultar_base_datos del proyecto principal añadiendo
PROJECT_ROOT a sys.path de forma robusta.
"""

import sys
import logging
from pathlib import Path
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# ── Añadir PROJECT_ROOT a sys.path ──────────────────────────────────────────
def _add_project_root() -> Path:
    """
    Sube desde este archivo hasta encontrar la carpeta que contiene 'src/'
    y la añade a sys.path. Retorna la ruta PROJECT_ROOT.
    """
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "src").is_dir():
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
                logger.debug("PROJECT_ROOT añadido: %s", root)
            return candidate
    raise RuntimeError(
        "No se encontró 'src/' en ningún directorio padre. "
        f"Archivo actual: {__file__}"
    )

PROJECT_ROOT = _add_project_root()

# ── Import del proyecto (igual que notebook 02) ──────────────────────────────
try:
    from src.entities.consultar_base_datos import consultar_base_datos
    _MONGO_OK = True
except Exception as e:
    logger.warning("MongoDB no disponible: %s", e)
    _MONGO_OK = False


# ── Cache del DataFrame completo ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_corpus_df() -> pd.DataFrame:
    """
    Carga el corpus completo desde MongoDB Atlas y lo cachea en memoria.
    Si MongoDB no está disponible, retorna un DataFrame vacío.

    Columnas esperadas: id, titulo, artista, genero, anio, letra,
                        idioma, fuente, url_fuente, metricas_*
    """
    if not _MONGO_OK:
        logger.error("MongoDB no disponible — retornando DataFrame vacío.")
        return pd.DataFrame()

    logger.info("Cargando corpus desde MongoDB Atlas…")
    try:
        cdb = consultar_base_datos()
        df  = cdb.cargar_todas().df
        logger.info("Corpus cargado: %d canciones.", len(df))
        return df
    except Exception as e:
        logger.error("Error al cargar corpus: %s", e)
        return pd.DataFrame()


def get_generos() -> list[str]:
    """Retorna la lista ordenada de géneros únicos del corpus."""
    df = get_corpus_df()
    if df.empty or "genero" not in df.columns:
        return []
    return sorted(df["genero"].dropna().unique().tolist())


def buscar_por_palabra(
    palabra: str,
    top_n: int = 10,
    genero: str | None = None,
) -> pd.DataFrame:
    """
    Busca canciones cuya letra contiene la palabra dada.
    Retorna columnas: titulo, artista, genero, anio, fragmento, ocurrencias.

    Parámetros
    ----------
    palabra : str   Palabra o frase a buscar (case-insensitive)
    top_n   : int   Máximo de resultados
    genero  : str   Filtrar por género (opcional)
    """
    df = get_corpus_df()
    if df.empty:
        return pd.DataFrame()

    mask = df["letra"].str.contains(palabra, case=False, na=False, regex=False)
    if genero:
        mask &= df["genero"] == genero

    result = df[mask].copy()

    # Contar ocurrencias
    result["ocurrencias"] = result["letra"].str.lower().str.count(palabra.lower())

    # Fragmento con contexto
    def _fragmento(letra: str, palabra: str, context: int = 80) -> str:
        pos = letra.lower().find(palabra.lower())
        if pos == -1:
            return letra[:120] + "…"
        start = max(0, pos - context)
        end   = min(len(letra), pos + len(palabra) + context)
        frag  = ("…" if start > 0 else "") + letra[start:end] + ("…" if end < len(letra) else "")
        return frag

    result["fragmento"] = result["letra"].apply(lambda x: _fragmento(str(x), palabra))
    result = result.sort_values("ocurrencias", ascending=False).head(top_n)

    cols = ["titulo", "artista", "genero", "anio", "ocurrencias", "fragmento"]
    return result[[c for c in cols if c in result.columns]].reset_index(drop=True)


def get_stats() -> dict:
    """Estadísticas generales del corpus para la página de inicio."""
    df = get_corpus_df()
    if df.empty:
        return {}

    stats = {
        "total":    len(df),
        "generos":  df["genero"].nunique() if "genero" in df.columns else 0,
        "artistas": df["artista"].nunique() if "artista" in df.columns else 0,
        "anio_min": int(df["anio"].min()) if "anio" in df.columns else "—",
        "anio_max": int(df["anio"].max()) if "anio" in df.columns else "—",
    }

    if "metricas_num_palabras" in df.columns:
        stats["avg_palabras"] = round(df["metricas_num_palabras"].mean(), 0)

    return stats
