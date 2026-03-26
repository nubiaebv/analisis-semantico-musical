"""
Clase: pipeline_spacy

 Py con funciones para ejecutar el pos tagger con spacy


"""
import sys

from src.utils.path import abrir_archivo
# Importar todas las librerías necesarias
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✓ Librerías importadas correctamente")

class pipeline_spacy:
    def __init__(self, df = None):
        self._cargar_recursos_spacy()
        self._df = df

    def _cargar_recursos_spacy(self):
        # Descargar recursos necesarios de Spacy (si no están ya instalados)

        print("Cargando recursos de Spacy...\n")

        # Cargar modelo de Spacy en inglés
        print("Cargando modelo de Spacy...")
        try:
            self._nlp = spacy.load("en_core_web_sm")
            print("✓ Modelo de Spacy cargado correctamente")
        except OSError:
            print("⚠ Modelo no encontrado. Instalando...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self._nlp = spacy.load("en_core_web_sm")
            print("✓ Modelo de Spacy instalado y cargado")

        print("\n" + "=" * 60)
        print("¡Listo para comenzar con el POS Tagging!")
        print("=" * 60)
        # Tokenización

    def _realizar_token(self, letra):
        doc = self._nlp(letra)
        token = []
        for tok in doc:
            tokens = tok.text
            token.append(tokens)
        return token

    def _paso_tokenizacion(self):
        tqdm.pandas(desc="Paso 1 Tokenización")
        self._df['tokens'] = self._df['letra_cancion'].progress_apply(self._realizar_token)

        # Etiquetado POS

    def _realizar_etiquetado(self, tokens_lista):
        """
        Recibe una lista de tokens (strings) y devuelve tuplas (token, tag)
        """
        # Reconstruir el texto y procesarlo con spaCy
        texto = " ".join(tokens_lista)
        doc = self._nlp(texto)
        etiquetas = [(token.text, token.pos_) for token in doc]
        return etiquetas

    def _paso_pos_tagging(self):
        tqdm.pandas(desc="Paso 2: Etiquetado POS")
        self._df['Etiquetado_POS'] = self._df['tokens'].progress_apply(self._realizar_etiquetado)

        # Borrado de StopWords y NER

    def _eliminar_stopwords(self, etiquetas_pos):
        """
        Recibe una lista de tuplas (token, tag) y elimina las stopwords
        """
        sin_stopwords = [(token, tag) for token, tag in etiquetas_pos
                         if token.lower() not in self._nlp.Defaults.stop_words]
        return sin_stopwords

    def _paso_stopwords(self):
        tqdm.pandas(desc="Paso 3: Eliminar Stopwords")
        self._df['StopWords'] = self._df['Etiquetado_POS'].progress_apply(self._eliminar_stopwords)

        # Mayúsculas y minúsculas

    def _aplicar_minusculas(self, etiquetas_pos):
        """
        Recibe lista de tuplas (token, tag) y convierte tokens a minúsculas
        """
        minusculas = [(token.lower(), tag) for token, tag in etiquetas_pos]
        return minusculas

    def _paso_minusculas(self):
        tqdm.pandas(desc="Paso 4: Aplicar Minúsculas")
        self._df['Minusculas'] = self._df['StopWords'].progress_apply(self._aplicar_minusculas)

        # Lematización

    def _aplicar_lematizacion(self, tuplas_tokens):
        """
        Recibe lista de tuplas (token, tag) y devuelve (lemma, tag)
        """
        # Reconstruir el texto desde las tuplas
        texto = " ".join([token for token, tag in tuplas_tokens])
        doc = self._nlp(texto)

        # Obtener lemmas con sus tags
        lemas = [(token.lemma_, token.pos_) for token in doc]
        return lemas

    def _paso_lematizacion(self):
        tqdm.pandas(desc="Paso 5: Lematización")
        self._df['Lematizado'] = self._df['Minusculas'].progress_apply(self._aplicar_lematizacion)

    # Ejecutar pipeline completo

    def ejecutar(self):

        self._paso_tokenizacion()
        self._paso_pos_tagging()
        self._paso_stopwords()
        self._paso_minusculas()
        self._paso_lematizacion()

        return self._df