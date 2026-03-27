import re
import os
import pandas as pd
from langdetect import detect, LangDetectException

class clear_corpus:
    """
    Pipeline completo de limpieza del corpus de canciones.

    """

    # --- Constantes ---
    FRASES_BASURA = [
        r"lyrics for this song have yet to be released",
        r"come back soon when there is lyrics",
        r"no lyrics for this song",
        r"lyrics have yet to be released",
        r"users considering it'?s a virus or malware",
        r"^soon$",
        r"^lyrics$",
        r"^come back soon$",
    ]
    UNICODE_INVISIBLES = r"[\u200b\u200c\u200d\u00ad\ufeff]"

    def __init__(
            self,
            columna_lyric: str = "letra_cancion",
            columna_artist: str = "musico",
            columna_title: str = "nombre_cancion",
            columna_year: str = "Periodo",
            columna_genre: str = "Genero",
            umbral_palabras: int = 20,
            anio_min: int = 1990,
            anio_max: int = 2025,
            idioma_objetivo: str = "en",
    ):
        self.columna_lyric = columna_lyric
        self.columna_artist = columna_artist
        self.columna_title = columna_title
        self.columna_year = columna_year
        self.columna_genre = columna_genre
        self.umbral_palabras = umbral_palabras
        self.anio_min = anio_min
        self.anio_max = anio_max
        self.idioma_objetivo = idioma_objetivo

        self._patron_placeholder = re.compile(
            "|".join(self.FRASES_BASURA), flags=re.IGNORECASE
        )

    def _eliminar_placeholders(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df[self.columna_lyric].str.strip().str.match(
            self._patron_placeholder, na=False
        )
        print(f"[1] Placeholders encontrados: {mask.sum()}")
        return df[~mask].copy()

    def _eliminar_letras_cortas(self, df: pd.DataFrame) -> pd.DataFrame:
        df["_n_palabras"] = df[self.columna_lyric].str.split().str.len()
        mask = df["_n_palabras"] < self.umbral_palabras
        print(f"[2] Letras con menos de {self.umbral_palabras} palabras: {mask.sum()}")
        return df[~mask].drop(columns="_n_palabras").copy()

    def _filtrar_anios(self, df: pd.DataFrame) -> pd.DataFrame:
        mask_valido = df[self.columna_year].between(self.anio_min, self.anio_max)
        print(f"[3] Registros fuera de rango de años ({self.anio_min}–{self.anio_max}): {(~mask_valido).sum()}")
        return df[mask_valido].copy()

    def _limpiar_unicode(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df[self.columna_lyric].str.contains(
            self.UNICODE_INVISIBLES, regex=True, na=False
        )
        print(f"[4] Letras con caracteres Unicode invisibles: {mask.sum()}")
        df = df.copy()
        df[self.columna_lyric] = (
            df[self.columna_lyric]
            .str.replace(self.UNICODE_INVISIBLES, "", regex=True)
            .str.strip()
        )
        return df

    @staticmethod
    def _detectar_idioma(texto: str) -> str:
        try:
            return detect(str(texto)[:500])
        except LangDetectException:
            return "unknown"

    def _filtrar_idioma(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["idioma"] = df[self.columna_lyric].apply(self._detectar_idioma)
        print(f"[5] Distribución de idiomas:\n{df['idioma'].value_counts().to_string()}")
        #df_filtrado = df[df["idioma"] == self.idioma_objetivo].copy()
        #print(f"    Registros en '{self.idioma_objetivo}': {len(df_filtrado):,}")
        return df

    def _normalizar_texto(self, texto):
        if not isinstance(texto, str): return "desconocido"
        # 1. Todo a minúsculas
        texto = texto.lower()
        # 2. Reemplazar guiones, barras y puntos por espacios
        texto = re.sub(r'[-/.]', ' ', texto)
        # 3. Quitar caracteres especiales residuales (letras y espacios)
        texto = re.sub(r'[^a-z\s]', '', texto)
        # 4. Colapsar espacios múltiples
        texto = ' '.join(texto.split())
        return texto

    def _normaliza_genero(self, df):

        col = self.columna_genre
        df[col] = df[col].apply(self._normalizar_texto)
        return df

    def _limpiar_estructura_cancion(self, texto):
        if not isinstance(texto, str): return ""
        # 1. Eliminar corchetes [Intro], [Verso], etc.
        texto = re.sub(r'\[.*?\]', '', texto)
        # 2. Eliminar "Letra de ..."
        texto = re.sub(r'letra de ".*?"', '', texto, flags=re.IGNORECASE)
        # 3. Limpiar saltos de línea y espacios
        texto = re.sub(r'\n+', ' ', texto)
        texto = ' '.join(texto.split())
        return texto.strip()

    def _normaliza_letra(self, df):

        col = self.columna_lyric
        df[col] = df[col].apply(self._limpiar_estructura_cancion)
        return df

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def limpiar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo y retorna el DataFrame limpio.

        """
        print(f"Buscando la columna: '{self.columna_lyric}'")
        print(f"¿Está en el DF?: {self.columna_lyric in df.columns}")
        print(f"Registros iniciales: {len(df):,}\n")

        df = self._eliminar_placeholders(df)
        df = self._eliminar_letras_cortas(df)
        df = self._filtrar_anios(df)
        df = self._limpiar_unicode(df)
        df = self._normaliza_genero(df)
        df = self._normaliza_letra(df)
        df = self._filtrar_idioma(df)

        print(f"\nRegistros finales: {len(df):,}")
        return df