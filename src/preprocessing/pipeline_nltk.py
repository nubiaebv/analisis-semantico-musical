"""
Clase: pipeline_nltk

Objetivo: Py con funciones para ejecutar el pos tagger con nltk

Cambios:

"""
# Configurar SSL PRIMERO (antes de importar NLTK)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ahora importar todas las librerías necesarias
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import warnings

warnings.filterwarnings('ignore')

class pipeline_nltk:

    def __init__(self, df:None):

        self._cargar_recursos_nltk()
        self._df = df



    def _cargar_recursos_nltk(self):
        print("Cargando recursos de NLTK ...\n")

        # Intentar descargar recursos de NLTK de forma silenciosa
        try:
            nltk.data.find('tokenizers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)

        try:
            nltk.data.find('taggers/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            nltk.data.find('taggers/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        print("✓ Recursos de NLTK listos")
        print("\n" + "=" * 60)
        print("¡Listo para comenzar con el POS Tagging!")
        print("=" * 60)

    # Paso 1 Tokenización
    def _realizar_token(self, letra):
        sentences = sent_tokenize(letra)
        token = []
        for sent in sentences:
            tokens = word_tokenize(sent)
            token.append(tokens)
        return token

    def _paso_tokenizacion(self):
        tqdm.pandas(desc="Paso 1 Tokenización")
        self._df['tokens'] = self._df['letra_cancion'].progress_apply(self._realizar_token)

    # Paso 2 Etiquetado POS
    def _realizar_taggins(self, token):
        sentences = token
        analisis = []
        for sent in sentences:
            pos_tags = pos_tag(sent)
            analisis.append(pos_tags)
        return analisis

    def _paso_pos_tagging(self):
        tqdm.pandas(desc="Paso 2 Etiquetado POS")
        self._df['Etiquetado_POS'] = self._df['tokens'].progress_apply(self._realizar_taggins)

    # Paso 3 Borrado de StopWords
    def _borrado_stopWords(self, pos_tags_list):
        """Elimina stopwords comunes"""
        stop_words = set(stopwords.words('english'))
        resultado = []
        for sentence_tags in pos_tags_list:
            sentence_clean = [(word, tag) for word, tag in sentence_tags
                              if word not in stop_words]
            if sentence_clean:
                resultado.append(sentence_clean)
        return resultado

    def _paso_stopwords(self):
        tqdm.pandas(desc="Paso 3 Borrado de StopWords")
        self._df['StopWords'] = self._df['Etiquetado_POS'].progress_apply(self._borrado_stopWords)

    # Paso 4 Mayúsculas / minúsculas
    def _convertir_minusculas(self, pos_tags_list):
        """Convierte todos los tokens a minúsculas"""
        resultado = []
        for sentence_tags in pos_tags_list:
            sentence_lower = [(word.lower(), tag) for word, tag in sentence_tags]
            resultado.append(sentence_lower)
        return resultado

    def _paso_minusculas(self):
        tqdm.pandas(desc=" Paso 4 Mayúsculas / minúsculas")
        self._df['pos_tags_lower'] = self._df['StopWords'].progress_apply(self._convertir_minusculas)

    # Paso 5 Lematización
    def _get_wordnet_pos(self, tag):
        """Convierte POS tag de NLTK a formato WordNet"""
        if tag.startswith('J'):
            return 'a'  # Adjetivo
        elif tag.startswith('V'):
            return 'v'  # Verbo
        elif tag.startswith('N'):
            return 'n'  # Sustantivo
        elif tag.startswith('R'):
            return 'r'  # Adverbio
        else:
            return 'n'  # Default: sustantivo

    def _lematizar(self, pos_tags_list):
        """Aplica lematización usando POS tags"""
        lemmatizer = WordNetLemmatizer()
        resultado = []
        for sentence_tags in pos_tags_list:
            sentence_lemma = []
            for word, tag in sentence_tags:
                wordnet_pos = self._get_wordnet_pos(tag)
                lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
                sentence_lemma.append((lemma, tag))
            resultado.append(sentence_lemma)
        return resultado

    def _paso_lematizacion(self):
        tqdm.pandas(desc="Paso 5 Lematización")
        self._df['Lematizado'] = self._df['pos_tags_lower'].progress_apply(self._lematizar)

    # Ejecutar pipeline completo
    def ejecutar(self):
        self._paso_tokenizacion()
        self._paso_pos_tagging()
        self._paso_stopwords()
        self._paso_minusculas()
        self._paso_lematizacion()

        return self._df