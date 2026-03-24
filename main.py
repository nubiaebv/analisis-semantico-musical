from src.preprocessing.GeniusScraper import GeniusScraper, GENIUS_TOKEN, ARTISTAS
from src.preprocessing.preprocessing_corpus import preprocessing_corpus


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



