import pandas as pd

from src.utils.path import abrir_archivo
from src.preprocessing.clear_corpus import clear_corpus
from src.preprocessing.pipeline_nltk import pipeline_nltk
from src.preprocessing.pipeline_spacy import pipeline_spacy
from src.entities.insertar_base_datos import insertar_base_datos

class preprocessing_corpus:
    def __init__(self, archivo=True):
        self._clear_corpus = clear_corpus()
        if archivo:
            self._df = abrir_archivo()
        else:
            self._df = abrir_archivo("corpus_canciones_webScraping.csv")

        print(len(self._df))


    def procesar_corpus(self):
        print("Procesando corpus limpieza")
        df = self._clear_corpus.limpiar(self._df)
        nltk = pipeline_nltk(df)
        spacy = pipeline_spacy(df)
        if df is not None:
            print("Procesando corpus spacy")
            df_spacy= spacy.ejecutar()
            #print(df_spacy.info())
            df = pd.merge(
                df,
                df_spacy[['Artist', 'nombre_cancion', 'letra_cancion','Periodo','Genero','Lematizado']],  # Solo tomamos las llaves y la que quieres agregar
                on=['Artist', 'nombre_cancion', 'letra_cancion','Periodo','Genero'],
                how='left'
            )
            df.rename(
                columns={'Lematizado_y': 'Lematizado_Spacy'}, inplace=True)
        #print(df.info())
        if df is not None:
            print("Procesando corpus nltk")
            df_nltk = nltk.ejecutar()
            #print(df_nltk.info())
            df = pd.merge(
                df,
                df_nltk[['Artist', 'nombre_cancion', 'letra_cancion', 'Periodo', 'Genero', 'Lematizado']],
                # Solo tomamos las llaves y la que quieres agregar
                on=['Artist', 'nombre_cancion', 'letra_cancion', 'Periodo', 'Genero'],
                how='left'
            )
            df.rename(
                columns={'Lematizado': 'Lematizado_nltk'}, inplace=True)
        #print(df.info())
        if df is not None:
            print("Procesando corpus en mongo")
            insertor = insertar_base_datos(batch_size=100)
            insertor.insertar(df)

        return df



