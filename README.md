# Análisis Semántico de Letras Musicales

> Aplicando **Word2Vec**, **BETO** y **MongoDB** para descubrir relaciones semánticas profundas en letras musicales, comparando representaciones vectoriales estáticas versus contextuales a través de un pipeline completo de NLP.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-4.x-ED8B00)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BETO-FFD21E?logo=huggingface&logoColor=black)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-2.x-3F4F75?logo=plotly&logoColor=white)
![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-yellow)


---

##  Tabla de Contenidos

- [Descripción](#-descripción)
- [Objetivos](#-objetivos)
- [Tecnologías](#-tecnologías)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Instalación](#️-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Pipeline](#-pipeline)
- [Análisis Implementados](#-análisis-implementados)
- [Autores](#-autores)

---

##  Descripción

Este proyecto extiende el **Proyecto 1** (Análisis Morfosintáctico con POS Tagging) incorporando técnicas avanzadas de representación semántica. Se entrena **Word2Vec** (CBOW y Skip-Gram) sobre un corpus musical enriquecido mediante **Web Scraping**, y se aplica **BETO** (Spanish BERT) para generar embeddings contextuales. Todo el corpus se gestiona en **MongoDB Atlas** y los resultados se exploran en un dashboard interactivo con **Plotly Dash**.

| Aspecto | Detalle |
|---|---|
| **Curso** | Minería de Textos — CUC |
| **Prerequisito** | Proyecto 1 completado (corpus con POS Tagging) |
| **Corpus mínimo** | 6,000 – 12,000 canciones (Proyecto 1 + scraping) |
| **Base de datos** | MongoDB Atlas (free tier) |

---

##  Objetivos

- Implementar un scraper funcional que enriquezca el corpus con al menos 1,000 canciones nuevas respetando buenas prácticas (rate limiting, `robots.txt`).
- Diseñar un esquema documental en MongoDB, migrar los datos del Proyecto 1 e integrar los nuevos datos del scraping.
- Entrenar modelos **Word2Vec** (CBOW y Skip-Gram) para descubrir campos semánticos, realizar analogías vectoriales y comparar géneros musicales.
- Utilizar **BETO** desde HuggingFace para generar embeddings contextuales y demostrar diferencias frente a representaciones estáticas.
- Comparar cuantitativamente **BoW vs. Word2Vec vs. BETO** en clasificación, clustering y proyecciones t-SNE.
- Presentar todos los hallazgos en un dashboard interactivo con **Plotly Dash**.

---

##  Tecnologías

| Herramienta | Versión | Propósito |
|---|---|---|
| Python | 3.9+ | Lenguaje base |
| MongoDB / pymongo | Atlas · 4.x | Almacenamiento NoSQL del corpus |
| Gensim | 4.x | Entrenamiento Word2Vec |
| Transformers (HuggingFace) | 4.x | Carga y uso de BETO |
| PyTorch | 2.x | Backend para BETO |
| scikit-learn | 1.x | BoW, TF-IDF, KMeans, t-SNE |
| BeautifulSoup4 | 4.x | Parsing HTML para scraping |
| requests | 2.x | HTTP requests para scraping |
| spaCy | 3.x | Preprocesamiento de texto |
| NLTK | 3.x | Tokenización y stopwords |
| Pandas | 2.x | Manipulación de datos |
| Matplotlib / Seaborn | — | Visualización estática |
| Plotly / Dash | 5.x / 2.x | Dashboard interactivo |
| Jupyter Notebook | — | Desarrollo exploratorio |

---

##  Estructura del Repositorio

```
analisis-semantico-musical/
│
├── dashboard/                        # Dashboard analítico con Plotly Dash
│   ├── assets/
│   │   └── style.css                 # Estilos personalizados
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── beto.py                   # Vista: análisis de embeddings BETO
│   │   ├── bow_tfidf.py              # Vista: representaciones BoW / TF-IDF
│   │   ├── busqueda.py               # Vista: búsqueda semántica de canciones
│   │   ├── comparacion.py            # Vista: comparación entre representaciones
│   │   └── word2vec.py               # Vista: análisis Word2Vec
│   ├── __init__.py
│   ├── app.py                        # Punto de entrada del dashboard
│   ├── components.py                 # Componentes reutilizables de UI
│   └── db.py                         # Conexión a MongoDB desde el dashboard
│
├── data/
│   ├── processed/
│   │   ├── corpus_canciones.csv          # Corpus base del Proyecto 1
│   │   └── corpus_canciones_webScraping.csv  # Canciones nuevas por scraping
│   ├── raw/                              # Archivos originales sin procesar
│   └── results/
│       ├── w2v_cbow.model                # Modelo Word2Vec CBOW entrenado
│       └── w2v_skipgram.model            # Modelo Word2Vec Skip-Gram entrenado
│
├── docs/                                 # Imágenes y documentación generada
│
├── notebooks/
│   ├── 03_word2vec_analisis.ipynb        # Entrenamiento y análisis Word2Vec
│   ├── 04_beto_analisis.ipynb            # Embeddings contextuales con BETO
│   └── 05_comparacion_final.ipynb        # Comparación BoW vs W2V vs BETO
│
├── src/
│   ├── database/
│   │   ├── __init__.py
│   │   └── MongoConnection.py            # Conexión y operaciones MongoDB
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embeddings_beto.py            # Generación de embeddings con BETO
│   │   └── embeddings_w2v.py             # Entrenamiento y uso de Word2Vec
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── CancionEntity.py              # Modelo/entidad de canción
│   │   ├── consultar_base_datos.py       # Consultas al corpus en MongoDB
│   │   └── insertar_base_datos.py        # Inserción de documentos
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clear_corpus.py               # Limpieza del corpus
│   │   ├── GeniusScraper.py              # Scraper de letras musicales
│   │   ├── pipeline_nltk.py              # Preprocesamiento con NLTK
│   │   ├── pipeline_spacy.py             # Preprocesamiento con spaCy
│   │   └── preprocessing_corpus.py       # Pipeline de preprocesamiento general
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── CancionRepository.py          # Repositorio de acceso a datos
│   ├── services/
│   │   ├── __init__.py
│   │   └── CancionService.py             # Lógica de negocio
│   ├── utils/
│   │   ├── __init__.py
│   │   └── path.py                       # Utilidades de rutas del proyecto
│   └── __init__.py
│
├── .env                              # Variables de entorno (no versionado)
├── .gitignore
├── main.py                           # Punto de entrada principal
├── README.md
├── USO_DE_IA.md                      # Registro de uso de herramientas de IA
└── requirements.txt
```

---

##  Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/nubiaebv/analisis-semantico-musical.git
cd analisis-semantico-musical
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar modelos de NLP

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

```bash
python -m spacy download es_core_news_sm   # Español
python -m spacy download en_core_web_sm    # Inglés
```

---

##  Configuración

Crea un archivo `.env` en la raíz del proyecto con tus credenciales de MongoDB Atlas:

```env
MONGO_URI=mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/
MONGO_DB_NAME=analisis_musical
```

>  El archivo `.env` está incluido en `.gitignore` y **nunca debe subirse al repositorio**.

---

##  Uso

### Ejecutar los notebooks

```bash
jupyter notebook
```

Navega a `notebooks/` y ejecuta los archivos en orden numérico para reproducir el pipeline completo.

### Ejecutar el dashboard

```bash
python dashboard/app.py
```

Abre tu navegador en [http://127.0.0.1:8050](http://127.0.0.1:8050) para explorar el dashboard interactivo.

---

##  Pipeline

```
Corpus Proyecto 1 (CSV)
        │
        ▼
 Migración a MongoDB  ──►  Web Scraping (+1,000 canciones)
        │                          │
        └──────────┬───────────────┘
                   ▼
          Preprocesamiento
        (NLTK · spaCy · limpieza)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    BoW/TF-IDF  Word2Vec    BETO
   (scikit-learn) (gensim) (transformers)
        │          │          │
        └──────────┴──────────┘
                   ▼
         Análisis y Comparación
                   │
                   ▼
         Dashboard Plotly Dash
```

---

##  Análisis Implementados

### Word2Vec — `03_word2vec_analisis.ipynb`

- **Campos semánticos por género** — modelo entrenado por género con exploración de vecinos semánticos via `most_similar()`.
- **Analogías vectoriales** — al menos 5 analogías en el dominio musical (ej: `rock − guitarra + piano = ?`).
- **Similitud entre géneros** — vector centroide por género y distancia coseno entre ellos.
- **Vocabulario exclusivo** — palabras más características de cada género por posición en el espacio vectorial.
- **Visualización t-SNE** — proyección 2D de las palabras más frecuentes del corpus con agrupación automática por KMeans.

### BETO — `04_beto_analisis.ipynb`

- **Polisemia contextual** — embeddings distintos para una misma palabra según el género (ej: "fuego", "noche", "camino").
- **Búsqueda semántica** — dado un texto de consulta, recupera las canciones más similares usando el vector `[CLS]`.
- **Masked Language Model** — análisis de cómo BETO completa frases típicas de cada género musical.

### Comparación BoW vs. Word2Vec vs. BETO — `05_comparacion_final.ipynb`

- **Clasificación de género** — Logistic Regression / KNN con cada representación, comparando accuracy.
- **Clustering** — K-Means evaluado con Silhouette Score por tipo de representación.
- **Visualización t-SNE comparativa** — proyecciones 2D de documentos con cada representación para comparar separación visual entre géneros.

---

##  Autores

**Nubia Elena Brenes Valerín** · **Pablo Andrés Marín Castillo**

---

<p align="center">
  Proyecto desarrollado para el curso <strong>Minería de Textos</strong> · Diplomado en Big Data<br>
  <strong>Colegio Universitario de Cartago (CUC)</strong>
</p>

