from src.preprocessing.GeniusScraper import GeniusScraper, GENIUS_TOKEN, ARTISTAS
from src.preprocessing.preprocessing_corpus import preprocessing_corpus
from src.embeddings.embeddings_beto import CargadorBETO

ejecutar_webscraping_corpus = False

if ejecutar_webscraping_corpus:
    scraper = GeniusScraper(
        token=GENIUS_TOKEN,
        delay=1.5,
        checkpoint_file="data/processed/genius_dataset.csv",
    )
    df = scraper.run(
        artistas=ARTISTAS,
        max_songs=40,
    )

    print("\nPreview del dataset:")
    print(df.head(5).to_string(index=False))

else:
    print("Procesar Archivo local")
    corpus = preprocessing_corpus(True)
    df = corpus.procesar_corpus()
    print(df.head())
    print(df.info())
    print("Procesar Archivo Scraper ")
    corpus_scraper = preprocessing_corpus(False)
    df =corpus_scraper.procesar_corpus()
    print("Procesar word2vec")

    cargador = consultar_base_datos()
    cargador.cargar_por_generos(["pop", "alternative pop", "hip hop", "alternative rock", "dance pop", "rock"])
    df = cargador.df

    print(f"Canciones cargadas: {len(df):,}")
    print(f"Géneros disponibles: {df['genero'].unique()}")
    print(f"Idiomas: {df['idioma'].value_counts().to_dict()}")

    actualizar_embeddings_mongodb(
        df=df,
        modelo=entrenador.modelo_sg,
        col_id="id",
        col_letra="letra",
        batch_size=100,
    )
    print("word2vec_avg actualizado correctamente en MongoDB.")
    print("Procesar Beto")

    beto = CargadorBETO()
    tokenizer = beto.tokenizer
    model = beto.model

    actualizar_beto_cls_mongodb(
        df=df,
        tokenizer=tokenizer,
        model=model,
        col_id="id",
        col_letra="letra",
        batch_size=32,
    )
    print("beto_cls actualizado correctamente en MongoDB.")


