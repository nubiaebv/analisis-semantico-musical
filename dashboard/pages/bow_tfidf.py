"""
dashboard/pages/bow_tfidf.py
============================
Página: Análisis BoW / TF-IDF
"""

from dash import html, dcc, Input, Output, callback, register_page
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dashboard.components import (
    PLOTLY_LAYOUT, ACCENT1, ACCENT2, ACCENT3, BORDER,
    TEXT_PRI, TEXT_SEC, TEXT_MUTED, DARK_CARD, DARK_PANEL,
    FONT_MONO, empty_fig, genre_color_map,
    stat_card, section_header, page_header, info_box, card,
)
from dashboard.db import get_corpus_df, get_generos

register_page(__name__, path="/bow-tfidf", name="BoW / TF-IDF")

# ── Helper: convierte hex (#RRGGBB) a rgba(r,g,b,alpha) ─────────────────────
def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Cache de vectorización ───────────────────────────────────────────────────
_rep_cache: dict = {}

def _get_rep():
    if "rep" not in _rep_cache:
        df = get_corpus_df()
        if df.empty:
            return None, None, [], []
        textos  = df["letra"].fillna("").tolist()
        generos = df["genero"].fillna("unknown").tolist()
        vec = TfidfVectorizer(max_features=5000, min_df=2, stop_words="english")
        mat = vec.fit_transform(textos)
        _rep_cache["rep"]     = vec
        _rep_cache["mat"]     = mat
        _rep_cache["textos"]  = textos
        _rep_cache["generos"] = generos
    return (
        _rep_cache["rep"],
        _rep_cache["mat"],
        _rep_cache["textos"],
        _rep_cache["generos"],
    )


# ── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div([
    page_header(
        "Bag of Words / TF-IDF",
        "Representación dispersa — línea base del proyecto",
        badge="DISPERSO",
        badge_class="badge-sparse",
    ),

    # Stats
    html.Div(id="bow-stats-row", className="stat-grid"),

    # Fila 1: Top palabras por género + heatmap similitud
    html.Div([
        html.Div([
            card([
                section_header("Top palabras TF-IDF por género",
                               "Las palabras más discriminativas de cada género"),
                dcc.Dropdown(
                    id="bow-genero-sel",
                    placeholder="Selecciona un género…",
                    style={"background": DARK_CARD, "color": TEXT_PRI,
                           "fontSize": "12px", "marginBottom": "14px"},
                ),
                dcc.Graph(id="bow-top-words-chart",
                          config={"displayModeBar": False}, style={"height": "340px"}),
            ], accent="top"),
        ], style={"flex": "1", "minWidth": "300px"}),

        html.Div([
            card([
                section_header("Similitud coseno entre géneros",
                               "Vector promedio TF-IDF por género"),
                dcc.Graph(id="bow-heatmap",
                          config={"displayModeBar": False}, style={"height": "380px"}),
            ], accent="purple"),
        ], style={"flex": "1", "minWidth": "300px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),

    # Fila 2: Demo ortogonalidad + búsqueda TF-IDF
    html.Div([
        html.Div([
            card([
                section_header("Problema de ortogonalidad",
                               "BoW asigna similitud 0 a palabras semánticamente relacionadas"),
                info_box("💡 En BoW, 'love' y 'heart' tienen similitud coseno = 0.0, "
                         "aunque son semánticamente cercanas. Esto motiva Word2Vec y BERT."),
                dcc.Graph(id="bow-ortogonalidad-chart",
                          config={"displayModeBar": False}, style={"height": "260px"}),
            ], accent="amber"),
        ], style={"flex": "1", "minWidth": "280px"}),

        html.Div([
            card([
                section_header("Búsqueda semántica TF-IDF",
                               "Recupera canciones más similares a una consulta"),
                html.Div([
                    dcc.Input(
                        id="bow-query-input",
                        type="text",
                        placeholder="Escribe una consulta en inglés… ej: broken heart pain",
                        debounce=False,
                        className="search-input",
                        style={"marginBottom": "10px"},
                    ),
                    html.Button("🔍 Buscar", id="bow-btn-buscar",
                                className="btn-primary"),
                ], style={"marginBottom": "16px"}),
                html.Div(id="bow-search-results"),
            ], accent="top"),
        ], style={"flex": "1.2", "minWidth": "300px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginTop": "0"}),
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("bow-stats-row", "children"),
    Output("bow-genero-sel", "options"),
    Output("bow-genero-sel", "value"),
    Input("bow-stats-row", "id"),
)
def load_bow_stats(_):
    vec, mat, textos, generos = _get_rep()
    if vec is None:
        return [], [], None

    generos_unicos = sorted(set(generos))
    vocab_size = len(vec.vocabulary_)
    dispersi   = round((1 - mat.nnz / (mat.shape[0] * mat.shape[1])) * 100, 1)

    stats = html.Div([
        stat_card(f"{mat.shape[0]:,}", "Canciones", "corpus completo", ACCENT1, "🎵"),
        stat_card(f"{vocab_size:,}", "Vocabulario", "palabras únicas TF-IDF", ACCENT2, "📖"),
        stat_card(f"{dispersi}%", "Dispersión", "elementos = 0 en la matriz", ACCENT3, "⬜"),
        stat_card(f"{len(generos_unicos)}", "Géneros", "clases en el corpus", "#10B981", "🎸"),
    ], className="stat-grid")

    options = [{"label": g, "value": g} for g in generos_unicos]
    return stats, options, generos_unicos[0] if generos_unicos else None


@callback(
    Output("bow-top-words-chart", "figure"),
    Input("bow-genero-sel", "value"),
)
def update_top_words(genero):
    vec, mat, textos, generos = _get_rep()
    if vec is None or not genero:
        return empty_fig("Conecta MongoDB Atlas para ver los datos")

    indices = [i for i, g in enumerate(generos) if g == genero]
    if not indices:
        return empty_fig(f"No hay canciones para '{genero}'")

    vocab  = vec.get_feature_names_out()
    scores = mat[indices].mean(axis=0).A1
    top_n  = 15
    top_idx = scores.argsort()[-top_n:][::-1]
    words  = [vocab[i] for i in top_idx]
    vals   = [float(scores[i]) for i in top_idx]

    cmap  = genre_color_map(list(set(generos)))
    color = cmap.get(genero, ACCENT1)

    # FIX 1: usar rgba() en lugar de hex con alpha (#RRGGBBAA no es válido en Plotly)
    fig = go.Figure(go.Bar(
        y=words[::-1], x=vals[::-1],
        orientation="h",
        marker=dict(
            color=vals[::-1],
            colorscale=[[0, _hex_to_rgba(color, 0.25)], [1, color]],
        ),
        text=[f"{v:.4f}" for v in vals[::-1]],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_SEC, family=FONT_MONO),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=340)
    # FIX 3 (preventivo): separar ejes para evitar conflicto con PLOTLY_LAYOUT
    fig.update_xaxes(title="TF-IDF Score", gridcolor=BORDER)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


@callback(
    Output("bow-heatmap", "figure"),
    Input("bow-stats-row", "id"),
)
def update_heatmap(_):
    vec, mat, textos, generos = _get_rep()
    if vec is None:
        return empty_fig("Sin datos")

    generos_unicos = sorted(set(generos))
    cmap = genre_color_map(generos_unicos)

    genre_vecs = []
    for g in generos_unicos:
        idx = [i for i, gx in enumerate(generos) if gx == g]
        genre_vecs.append(mat[idx].mean(axis=0))

    sim = cosine_similarity(np.vstack([v.A for v in genre_vecs]))

    # FIX 2: reemplazar hex con alpha (#RRGGBBAA) por rgba() válido
    fig = go.Figure(go.Heatmap(
        z=sim, x=generos_unicos, y=generos_unicos,
        colorscale=[
            [0,   "#0A0E1A"],
            [0.5, _hex_to_rgba(ACCENT2, 0.53)],
            [1,   ACCENT1],
        ],
        zmin=0, zmax=1,
        text=np.round(sim, 3),
        texttemplate="%{text:.3f}",
        textfont=dict(size=10),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    # FIX 3 (preventivo): separar ejes
    fig.update_xaxes(tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)")
    return fig


@callback(
    Output("bow-ortogonalidad-chart", "figure"),
    Input("bow-stats-row", "id"),
)
def update_ortogonalidad(_):
    from sklearn.feature_extraction.text import CountVectorizer

    palabras = ["love", "heart", "guitar", "pain", "dance", "night"]
    mini = CountVectorizer()
    m    = mini.fit_transform(palabras)
    sim  = cosine_similarity(m)

    pairs = []
    vals  = []
    for i in range(len(palabras)):
        for j in range(i + 1, len(palabras)):
            pairs.append(f"{palabras[i]} × {palabras[j]}")
            vals.append(round(float(sim[i, j]), 3))

    colors = [ACCENT1 if v > 0 else TEXT_MUTED for v in vals]
    fig = go.Figure(go.Bar(
        x=pairs, y=vals,
        marker_color=colors,
        text=[str(v) for v in vals],
        textposition="outside",
        textfont=dict(size=10, family=FONT_MONO),
    ))
    # FIX 3: separar ejes del update_layout para evitar conflicto con PLOTLY_LAYOUT
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    fig.update_yaxes(range=[-0.1, 0.3], title="Similitud coseno", gridcolor=BORDER)
    fig.update_xaxes(tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)")
    fig.add_hline(y=0, line_color=ACCENT2, line_dash="dot", line_width=1)
    return fig


@callback(
    Output("bow-search-results", "children"),
    Input("bow-btn-buscar", "n_clicks"),
    Input("bow-query-input", "value"),
    prevent_initial_call=True,
)
def bow_search(n_clicks, query):
    from dash import ctx
    if not query or ctx.triggered_id != "bow-btn-buscar":
        return html.Div("Escribe una consulta y presiona Buscar.",
                        style={"color": TEXT_SEC, "fontSize": "12px", "padding": "10px 0"})

    vec, mat, textos, generos = _get_rep()
    if vec is None:
        return html.Div("MongoDB no disponible.", style={"color": "#EF4444"})

    q_vec = vec.transform([query.lower()])
    sims  = cosine_similarity(q_vec, mat).flatten()
    top5  = sims.argsort()[-5:][::-1]

    df = get_corpus_df()
    cmap = genre_color_map(list(set(generos)))

    rows = []
    for rank, idx in enumerate(top5, 1):
        row = df.iloc[idx]
        rows.append(html.Div([
            html.Div(f"#{rank}", className=f"result-rank {'top' if rank <= 3 else ''}"),
            html.Div([
                html.Div(str(row.get("titulo", "—")), className="result-title"),
                html.Div([
                    str(row.get("artista", "—")),
                    html.Span(
                        str(row.get("genero", "—")),
                        className="genre-tag",
                        style={"background": f"{cmap.get(str(row.get('genero','')), ACCENT1)}20",
                               "color": cmap.get(str(row.get("genero", "")), ACCENT1),
                               "marginLeft": "8px"},
                    ),
                ], className="result-meta", style={"display": "flex", "alignItems": "center"}),
            ], style={"flex": "1"}),
            html.Div(f"{sims[idx]:.4f}", className="result-score"),
        ], className="result-row"))

    return html.Div(rows) if rows else html.Div(
        "Sin resultados para esta consulta.",
        style={"color": TEXT_SEC, "fontSize": "12px"}
    )