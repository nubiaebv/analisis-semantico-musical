"""
dashboard/pages/comparacion.py
==============================
Página: Comparación Final — BoW vs Word2Vec vs BERT

Cambios respecto a la versión original:
  - _load_all() ya no carga archivos .npy/.npz externos.
    Ahora construye las matrices directamente desde MongoDB:
      • TF-IDF  → vectorizando en caliente con sklearn (igual que bow_tfidf.py)
      • Word2Vec → columna embeddings_word2vec_avg del DataFrame de MongoDB
      • BERT     → columna embeddings_beto_cls del DataFrame de MongoDB
  - Se elimina la dependencia de RESULTS / data/results/.
  - El resto de la lógica (clasificación, clustering, t-SNE, gráficos)
    permanece exactamente igual.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from dash import html, dcc, Input, Output, callback, register_page
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

from dashboard.components import (
    PLOTLY_LAYOUT, ACCENT1, ACCENT2, ACCENT3, BORDER,
    TEXT_PRI, TEXT_SEC, TEXT_MUTED, DARK_CARD, DARK_PANEL, FONT_MONO,
    empty_fig, genre_color_map,
    stat_card, section_header, page_header, info_box, card,
)
from dashboard.db import get_corpus_df, get_generos

register_page(__name__, path="/comparacion", name="Comparación Final")

# ── Helper: hex (#RRGGBB) → rgba(r,g,b,alpha) ────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Carga de representaciones desde MongoDB ──────────────────────────────────
_data_cache: dict = {}


def _load_all() -> dict:
    """
    Construye las tres representaciones directamente desde MongoDB.

    Retorna un dict con:
        "tfidf"        → scipy.sparse matrix  (n, vocab)
        "w2v"          → np.ndarray           (n, dim_w2v)
        "bert"         → np.ndarray           (n, 768)
        "genres_tfidf" → np.ndarray           (n,)  — etiquetas alineadas con tfidf
        "genres_w2v"   → np.ndarray           (n,)  — etiquetas alineadas con w2v
        "genres_bert"  → np.ndarray           (n,)  — etiquetas alineadas con bert
    """
    if _data_cache:
        return _data_cache

    df = get_corpus_df()
    if df.empty:
        return _data_cache

    # ── TF-IDF (toda la colección) ───────────────────────────────────────────
    try:
        textos  = df["letra"].fillna("").tolist()
        generos = df["genero"].fillna("unknown").tolist()
        vec = TfidfVectorizer(max_features=5000, min_df=2, stop_words="english")
        mat = vec.fit_transform(textos)
        _data_cache["tfidf"]        = mat
        _data_cache["genres_tfidf"] = np.array(generos)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Error construyendo TF-IDF: %s", e)

    # ── Word2Vec doc-embeddings (desde MongoDB) ──────────────────────────────
    if "embeddings_word2vec_avg" in df.columns:
        mask_w2v = df["embeddings_word2vec_avg"].apply(
            lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
        )
        if mask_w2v.any():
            try:
                _data_cache["w2v"] = np.array(
                    df.loc[mask_w2v, "embeddings_word2vec_avg"].tolist()
                )
                _data_cache["genres_w2v"] = np.array(
                    df.loc[mask_w2v, "genero"].fillna("unknown").tolist()
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).error("Error cargando W2V de MongoDB: %s", e)

    # ── BERT embeddings (desde MongoDB) ─────────────────────────────────────
    if "embeddings_beto_cls" in df.columns:
        mask_bert = df["embeddings_beto_cls"].apply(
            lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
        )
        if mask_bert.any():
            try:
                _data_cache["bert"] = np.array(
                    df.loc[mask_bert, "embeddings_beto_cls"].tolist()
                )
                _data_cache["genres_bert"] = np.array(
                    df.loc[mask_bert, "genero"].fillna("unknown").tolist()
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).error("Error cargando BERT de MongoDB: %s", e)

    return _data_cache


def _get_subset(key: str, n_max: int = 800):
    """
    Retorna (X_sub, g_sub) con máximo n_max muestras aleatorias.
    Ahora cada representación tiene su propio array de géneros alineado.
    """
    data = _load_all()
    X = data.get(key)
    if X is None:
        return None, None

    # Cada clave tiene su propio array de géneros alineado
    genres_key = f"genres_{key}"
    g = data.get(genres_key)
    if g is None:
        # Fallback a genres_tfidf (compatibilidad)
        g = data.get("genres_tfidf")
    if g is None:
        return None, None

    n = min(len(g), X.shape[0] if hasattr(X, "shape") else len(X), n_max)
    rng = np.random.default_rng(42)
    idx = rng.choice(min(X.shape[0], len(g)), n, replace=False)

    return X[idx], g[idx]


# ── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div([
    page_header(
        "Comparación Final",
        "BoW / TF-IDF  ·  Word2Vec  ·  BERT — análisis cuantitativo y visual",
        badge="COMPARATIVO",
        badge_class="badge-live",
    ),

    html.Div(id="comp-stats-row", className="stat-grid"),

    # Fila 1: Accuracy + Silhouette + tabla
    html.Div([
        html.Div([
            card([
                section_header("Clasificación de género",
                               "Logistic Regression — validación cruzada 5-fold"),
                dcc.Loading(
                    dcc.Graph(id="comp-accuracy-chart",
                              config={"displayModeBar": False}, style={"height": "300px"}),
                    color=ACCENT1,
                ),
            ], accent="top"),
        ], style={"flex": "1", "minWidth": "250px"}),

        html.Div([
            card([
                section_header("Clustering K-Means",
                               "Silhouette Score (coseno) — más alto es mejor"),
                dcc.Loading(
                    dcc.Graph(id="comp-silhouette-chart",
                              config={"displayModeBar": False}, style={"height": "300px"}),
                    color=ACCENT1,
                ),
            ], accent="purple"),
        ], style={"flex": "1", "minWidth": "250px"}),

        html.Div([
            card([
                section_header("Tabla de métricas"),
                html.Div(id="comp-metrics-table"),
            ], accent="amber"),
        ], style={"flex": "1", "minWidth": "250px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),

    # Fila 2: Matriz de confusión + t-SNE comparativo
    html.Div([
        html.Div([
            card([
                section_header("Matriz de confusión",
                               "Split 80/20 — Logistic Regression"),
                html.Div([
                    dcc.Dropdown(
                        id="comp-confusion-model",
                        options=[
                            {"label": "BoW / TF-IDF", "value": "tfidf"},
                            {"label": "Word2Vec",      "value": "w2v"},
                            {"label": "BERT",          "value": "bert"},
                        ],
                        value="tfidf",
                        style={"background": DARK_CARD, "fontSize": "12px",
                               "marginBottom": "12px"},
                    ),
                ]),
                dcc.Loading(
                    dcc.Graph(id="comp-confusion-chart",
                              config={"displayModeBar": False}, style={"height": "380px"}),
                    color=ACCENT1,
                ),
            ], accent="top"),
        ], style={"flex": "1", "minWidth": "280px"}),

        html.Div([
            card([
                section_header("t-SNE comparativo 3-en-1",
                               "Proyección 2D por representación — mismo subconjunto"),
                html.Div([
                    dcc.Checklist(
                        id="comp-tsne-genres",
                        inline=True,
                        labelStyle={"marginRight": "10px", "fontSize": "11px",
                                    "color": TEXT_SEC, "cursor": "pointer"},
                        style={"marginBottom": "10px"},
                    ),
                ]),
                dcc.Loading(
                    dcc.Graph(id="comp-tsne-chart",
                              config={"displayModeBar": False}, style={"height": "380px"}),
                    color=ACCENT1,
                ),
            ], accent="top"),
        ], style={"flex": "2", "minWidth": "380px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginTop": "0"}),

    # Fila 3: Conclusiones
    html.Div([
        card([
            section_header("Conclusiones del proyecto"),
            html.Div(id="comp-conclusions"),
        ], accent="top"),
    ]),
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("comp-stats-row",       "children"),
    Output("comp-accuracy-chart",  "figure"),
    Output("comp-silhouette-chart","figure"),
    Output("comp-metrics-table",   "children"),
    Output("comp-conclusions",     "children"),
    Output("comp-tsne-genres",     "options"),
    Output("comp-tsne-genres",     "value"),
    Input("comp-stats-row", "id"),
)
def load_comparison(_):
    data    = _load_all()
    generos = get_generos()
    cmap    = genre_color_map(generos)

    model_keys   = ["tfidf", "w2v", "bert"]
    modelos      = ["BoW / TF-IDF", "Word2Vec", "BERT"]
    model_colors = [ACCENT1, ACCENT2, ACCENT3]

    # ── Stats ────────────────────────────────────────────────────────────────
    n_tfidf = data["tfidf"].shape[0] if "tfidf" in data else 0
    n_w2v   = len(data["w2v"])       if "w2v"   in data else 0
    n_bert  = len(data["bert"])      if "bert"  in data else 0

    stats = html.Div([
        stat_card(f"{n_tfidf:,}",  "TF-IDF",   "canciones vectorizadas",   ACCENT1, "📦"),
        stat_card(f"{n_w2v:,}",    "Word2Vec",  "con vector en MongoDB",    ACCENT2, "⚡"),
        stat_card(f"{n_bert:,}",   "BERT",      "con vector en MongoDB",    ACCENT3, "🧠"),
        stat_card(f"{len(generos)}","Géneros",  "clases en el corpus",      "#10B981", "🎸"),
    ], className="stat-grid")

    # ── Clasificación (5-fold CV) ─────────────────────────────────────────────
    clf_results: dict = {}
    for key in model_keys:
        X, g = _get_subset(key, n_max=600)
        if X is None or g is None:
            continue
        try:
            le    = LabelEncoder()
            y_enc = le.fit_transform(g)
            X_d   = X.toarray() if sp.issparse(X) else X
            if sp.issparse(X):
                clf = LogisticRegression(max_iter=500, n_jobs=-1)
            else:
                clf = Pipeline([("sc", StandardScaler()),
                                ("clf", LogisticRegression(max_iter=500, n_jobs=-1))])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_d, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
            clf_results[key] = {"mean": scores.mean(), "std": scores.std()}
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("CV error %s: %s", key, e)

    # ── Silhouette (K-Means) ──────────────────────────────────────────────────
    sil_results: dict = {}
    for key in model_keys:
        X, g = _get_subset(key, n_max=400)
        if X is None or g is None:
            continue
        try:
            n_clusters = len(set(g))
            X_d = X.toarray() if sp.issparse(X) else X
            km  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(X_d)
            sil = silhouette_score(X_d, labels, metric="cosine", sample_size=min(500, len(X_d)))
            sil_results[key] = sil
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Silhouette error %s: %s", key, e)

    best_clf = max(clf_results, key=lambda k: clf_results[k]["mean"]) if clf_results else None
    best_sil = max(sil_results, key=sil_results.get) if sil_results else None

    # ── Gráficos de barra ─────────────────────────────────────────────────────
    def _bar_chart(values_dict, ylabel, _unused, yrange=None):
        labels = [m for k, m in zip(model_keys, modelos) if k in values_dict]
        vals   = [values_dict[k]["mean"] if isinstance(values_dict[k], dict)
                  else values_dict[k]
                  for k in model_keys if k in values_dict]
        errs   = [values_dict[k]["std"]  if isinstance(values_dict[k], dict)
                  else None
                  for k in model_keys if k in values_dict]
        colors = [c for k, c in zip(model_keys, model_colors) if k in values_dict]

        fig = go.Figure(go.Bar(
            x=labels, y=vals,
            marker_color=colors,
            error_y=dict(type="data", array=[e for e in errs if e is not None],
                         color=TEXT_SEC, thickness=1.5) if any(e for e in errs) else None,
            text=[f"{v:.1%}" if isinstance(v, float) and v < 2 else f"{v:.3f}"
                  for v in vals],
            textposition="outside",
            textfont=dict(size=11, family=FONT_MONO),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        fig.update_yaxes(title=ylabel, range=yrange, gridcolor=BORDER)
        fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
        return fig

    fig_acc = _bar_chart(clf_results, "Accuracy (5-fold CV)", {})
    fig_sil = _bar_chart(sil_results, "Silhouette Score", {}, yrange=[0, 1])

    # ── Tabla de métricas ─────────────────────────────────────────────────────
    rows_html = []
    for key, label, color in zip(model_keys, modelos, model_colors):
        acc_val = (f"{clf_results[key]['mean']:.4f} ± {clf_results[key]['std']:.4f}"
                   if key in clf_results else "N/A")
        sil_val = f"{sil_results[key]:.4f}" if key in sil_results else "N/A"
        is_best_c = key == best_clf
        is_best_s = key == best_sil

        rows_html.append(html.Tr([
            html.Td(label,   style={"color": color, "fontWeight": "700",
                                    "padding": "10px 14px", "border": f"1px solid {BORDER}"}),
            html.Td(acc_val, style={"padding": "10px 14px", "border": f"1px solid {BORDER}",
                                    "fontFamily": FONT_MONO, "fontSize": "12px",
                                    "color": "#10B981" if is_best_c else TEXT_PRI}),
            html.Td(sil_val, style={"padding": "10px 14px", "border": f"1px solid {BORDER}",
                                    "fontFamily": FONT_MONO, "fontSize": "12px",
                                    "color": "#10B981" if is_best_s else TEXT_PRI}),
            html.Td("sparse" if key == "tfidf" else ("150d" if key == "w2v" else "768d"),
                    style={"padding": "10px 14px", "border": f"1px solid {BORDER}",
                           "color": TEXT_SEC, "fontFamily": FONT_MONO, "fontSize": "12px"}),
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={"background": DARK_PANEL, "color": TEXT_SEC, "fontSize": "10px",
                               "fontWeight": "700", "textTransform": "uppercase",
                               "letterSpacing": "1px", "padding": "10px 14px",
                               "border": f"1px solid {BORDER}"})
            for h in ["Modelo", "CV Accuracy", "Silhouette", "Dims"]
        ])),
        html.Tbody(rows_html),
    ], className="metrics-table")

    # ── Conclusiones ─────────────────────────────────────────────────────────
    conclusions = html.Div([
        html.Div([
            _conclusion_col("📦 BoW / TF-IDF", [
                "Representación dispersa — ~97% ceros",
                "Baseline sólido y fácilmente interpretable",
                "Ortogonalidad: 'love' ≠ 'heart' semánticamente",
                "No captura contexto ni relaciones de significado",
            ], ACCENT1),
            _conclusion_col("⚡ Word2Vec", [
                "Embeddings densos de 150 dimensiones",
                "Captura relaciones semánticas (love ↔ heart)",
                "Analogías vectoriales funcionan",
                "Un solo vector por palabra — sin contexto",
            ], ACCENT2),
            _conclusion_col("🧠 BERT", [
                "Embeddings contextuales de 768 dimensiones",
                "La misma palabra → vectores distintos por contexto",
                "Mejor rendimiento en clasificación y clustering",
                "Mayor costo computacional y de memoria",
            ], ACCENT3),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ])

    opts  = [{"label": g, "value": g} for g in generos]
    value = generos[:4] if generos else []
    return stats, fig_acc, fig_sil, table, conclusions, opts, value


def _conclusion_col(titulo, puntos, color):
    return html.Div([
        html.Div(titulo, style={"color": color, "fontWeight": "800", "fontSize": "13px",
                                 "marginBottom": "10px", "fontFamily": FONT_MONO}),
        html.Ul([
            html.Li(p, style={"color": TEXT_SEC, "fontSize": "12px",
                               "marginBottom": "4px", "lineHeight": "1.5"})
            for p in puntos
        ], style={"paddingLeft": "16px", "margin": "0"}),
    ], style={"flex": "1", "minWidth": "200px", "background": DARK_PANEL,
              "borderTop": f"3px solid {color}", "borderRadius": "8px",
              "padding": "16px"})


@callback(
    Output("comp-confusion-chart", "figure"),
    Input("comp-confusion-model", "value"),
)
def update_confusion(model_key):
    X, g = _get_subset(model_key, n_max=600)
    if X is None:
        return empty_fig(f"Datos no disponibles para {model_key}")
    try:
        le    = LabelEncoder()
        y_enc = le.fit_transform(g)
        X_d   = X.toarray() if sp.issparse(X) else X

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_d, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        if sp.issparse(X):
            clf = LogisticRegression(max_iter=500, n_jobs=-1)
        else:
            clf = Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=500, n_jobs=-1))])
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        labels = le.classes_
        cm     = confusion_matrix(y_te, y_pred)

        fig = go.Figure(go.Heatmap(
            z=cm, x=list(labels), y=list(labels),
            colorscale=[[0, "#0A0E1A"], [0.5, _hex_to_rgba(ACCENT2, 0.53)], [1, ACCENT1]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=10),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        fig.update_xaxes(title="Predicho", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9))
        fig.update_yaxes(title="Real",     gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9))
        return fig
    except Exception as e:
        return empty_fig(f"Error: {e}")


@callback(
    Output("comp-tsne-chart", "figure"),
    Input("comp-tsne-genres", "value"),
)
def update_tsne_comparison(generos_sel):
    if not generos_sel:
        return empty_fig("Selecciona géneros")

    data    = _load_all()
    generos = get_generos()
    cmap    = genre_color_map(generos)

    available = {
        k: n for k, n in
        [("tfidf", "BoW/TF-IDF"), ("w2v", "Word2Vec"), ("bert", "BERT")]
        if k in data
    }
    if not available:
        return empty_fig("No hay datos disponibles")

    n_cols = len(available)
    fig = make_subplots(rows=1, cols=n_cols,
                        subplot_titles=list(available.values()))

    for col_i, (key, label) in enumerate(available.items(), 1):
        X, g = _get_subset(key, n_max=400)
        if X is None:
            continue

        mask = np.isin(g, generos_sel)
        X_f, g_f = X[mask], g[mask]
        if len(g_f) < 10:
            continue

        X_d = X_f.toarray() if sp.issparse(X_f) else X_f
        n_s = min(300, len(X_d))
        idx = np.random.default_rng(42).choice(len(X_d), n_s, replace=False)
        X_s, g_s = X_d[idx], g_f[idx]

        tsne   = TSNE(n_components=2, perplexity=min(20, n_s - 1),
                      random_state=42, max_iter=300)
        coords = tsne.fit_transform(X_s)

        for g_name in sorted(set(g_s)):
            m = g_s == g_name
            fig.add_trace(go.Scatter(
                x=coords[m, 0], y=coords[m, 1], mode="markers",
                name=g_name, showlegend=(col_i == 1),
                marker=dict(color=cmap.get(g_name, ACCENT1), size=4, opacity=0.7),
            ), row=1, col=col_i)

    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    fig.update_layout(legend=dict(orientation="v", x=1.01, font=dict(size=9)))
    return fig