"""
dashboard/pages/word2vec.py
===========================
Página: Análisis Word2Vec

Cambios respecto a la versión original:
  - Word2VecService ahora se importa desde src.embeddings.embeddings_w2v_service
  - _get_doc_embeddings() lee los vectores word2vec_avg directamente de MongoDB
    (columna embeddings_word2vec_avg del DataFrame de consultar_base_datos)
    en lugar de cargar un archivo .npy externo.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dash import html, dcc, Input, Output, State, callback, register_page
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

from dashboard.components import (
    PLOTLY_LAYOUT, ACCENT1, ACCENT2, ACCENT3, BORDER,
    TEXT_PRI, TEXT_SEC, TEXT_MUTED, DARK_CARD, DARK_PANEL, FONT_MONO,
    empty_fig, genre_color_map,
    stat_card, section_header, page_header, info_box, card,
)
from dashboard.db import get_corpus_df, get_generos, PROJECT_ROOT

register_page(__name__, path="/word2vec", name="Word2Vec")

# ── Helper: hex (#RRGGBB) → rgba(r,g,b,alpha) ────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Carga del modelo Word2Vec ────────────────────────────────────────────────
_w2v_cache: dict = {}


def _get_w2v():
    """
    Carga los modelos Word2Vec desde data/results/ usando Word2VecService.
    Si los archivos no existen los entrena on-the-fly con el corpus de MongoDB.
    """
    if "svc" in _w2v_cache:
        return _w2v_cache["svc"]

    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        # ← nuevo import: adaptador que usa el backend real
        from src.embeddings.embeddings_w2v_service import Word2VecService
        models_dir = PROJECT_ROOT / "data" / "results"
        svc = Word2VecService()
        svc.load(models_dir, prefix="w2v")
        _w2v_cache["svc"] = svc
        return svc
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("No se pudo cargar Word2VecService: %s", e)
        _w2v_cache["svc"] = None
        return None


def _get_doc_embeddings() -> np.ndarray:
    """
    Lee los vectores word2vec_avg directamente del DataFrame de MongoDB.
    Retorna un np.ndarray de forma (n_canciones, dim) o None si no hay vectores.
    """
    df = get_corpus_df()
    if df.empty or "embeddings_word2vec_avg" not in df.columns:
        return None

    # Filtrar filas con vectores válidos (listas no vacías)
    mask = df["embeddings_word2vec_avg"].apply(
        lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
    )
    if not mask.any():
        return None

    try:
        return np.array(df.loc[mask, "embeddings_word2vec_avg"].tolist())
    except Exception:
        return None


# ── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div([
    page_header(
        "Word2Vec",
        "Embeddings estáticos — CBOW y Skip-Gram entrenados sobre el corpus",
        badge="DENSO ESTÁTICO",
        badge_class="badge-dense",
    ),

    html.Div(id="w2v-stats-row", className="stat-grid"),

    # Fila 1: Vecinos semánticos + analogías
    html.Div([
        html.Div([
            card([
                section_header("Vecinos semánticos", "Palabras más cercanas en el espacio vectorial"),
                html.Div([
                    dcc.Input(
                        id="w2v-word-input",
                        type="text",
                        placeholder="Escribe una palabra… ej: love, dance, pain",
                        className="search-input",
                        style={"marginBottom": "10px"},
                    ),
                    html.Div([
                        dcc.RadioItems(
                            id="w2v-model-sel",
                            options=[
                                {"label": "Skip-Gram", "value": "skipgram"},
                                {"label": "CBOW",      "value": "cbow"},
                            ],
                            value="skipgram",
                            inline=True,
                            labelStyle={"marginRight": "16px", "fontSize": "12px",
                                        "color": TEXT_SEC, "cursor": "pointer"},
                        ),
                        html.Button("Buscar", id="w2v-btn-vecinos",
                                    className="btn-primary", style={"marginLeft": "auto"}),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                ]),
                dcc.Graph(id="w2v-vecinos-chart",
                          config={"displayModeBar": False}, style={"height": "320px"}),
            ], accent="top"),
        ], style={"flex": "1", "minWidth": "300px"}),

        html.Div([
            card([
                section_header("Analogías vectoriales",
                               "A − B + C ≈ ?  (propiedad clave de Word2Vec)"),
                info_box("Ejemplo: king − man + woman ≈ queen"),
                html.Div([
                    html.Div([
                        html.Label("A (restar)", style={"fontSize": "10px", "color": TEXT_SEC}),
                        dcc.Input(id="w2v-ana-a", type="text", value="man",
                                  className="search-input", style={"marginTop": "4px"}),
                    ], style={"flex": "1"}),
                    html.Div("−", style={"padding": "24px 8px 0", "color": ACCENT2, "fontSize": "18px"}),
                    html.Div([
                        html.Label("B (sumar)", style={"fontSize": "10px", "color": TEXT_SEC}),
                        dcc.Input(id="w2v-ana-b", type="text", value="king",
                                  className="search-input", style={"marginTop": "4px"}),
                    ], style={"flex": "1"}),
                    html.Div("+", style={"padding": "24px 8px 0", "color": ACCENT1, "fontSize": "18px"}),
                    html.Div([
                        html.Label("C (sumar)", style={"fontSize": "10px", "color": TEXT_SEC}),
                        dcc.Input(id="w2v-ana-c", type="text", value="woman",
                                  className="search-input", style={"marginTop": "4px"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "8px", "marginBottom": "12px"}),
                html.Button("Calcular analogía", id="w2v-btn-analogia",
                            className="btn-primary", style={"marginBottom": "14px"}),
                html.Div(id="w2v-analogia-result"),
            ], accent="purple"),
        ], style={"flex": "1", "minWidth": "280px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),

    # Fila 2: Similitud géneros + t-SNE
    html.Div([
        html.Div([
            card([
                section_header("Similitud entre géneros",
                               "Vectores promedio Word2Vec por género"),
                dcc.Graph(id="w2v-genre-sim-chart",
                          config={"displayModeBar": False}, style={"height": "360px"}),
            ], accent="amber"),
        ], style={"flex": "1", "minWidth": "280px"}),

        html.Div([
            card([
                section_header("Visualización t-SNE",
                               "Proyección 2D de embeddings de documento (desde MongoDB)"),
                html.Div([
                    dcc.Checklist(
                        id="w2v-tsne-genres",
                        inline=True,
                        labelStyle={"marginRight": "12px", "fontSize": "11px",
                                    "color": TEXT_SEC, "cursor": "pointer"},
                        style={"marginBottom": "12px"},
                    ),
                ]),
                dcc.Graph(id="w2v-tsne-chart",
                          config={"displayModeBar": False}, style={"height": "340px"}),
            ], accent="top"),
        ], style={"flex": "1.5", "minWidth": "340px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginTop": "0"}),
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("w2v-stats-row", "children"),
    Output("w2v-tsne-genres", "options"),
    Output("w2v-tsne-genres", "value"),
    Input("w2v-stats-row", "id"),
)
def load_w2v_stats(_):
    svc    = _get_w2v()
    generos = get_generos()
    cmap   = genre_color_map(generos)

    vocab  = svc.vocab_size  if svc else "—"
    dim    = svc.vector_size if svc else "—"
    models = "CBOW + Skip-Gram" if (svc and svc.cbow and svc.skipgram) else (
             "Skip-Gram" if (svc and svc.skipgram) else (
             "CBOW"      if (svc and svc.cbow)     else "No cargado"))

    # Contar canciones que tienen vector word2vec en MongoDB
    df = get_corpus_df()
    n_con_vector = 0
    if not df.empty and "embeddings_word2vec_avg" in df.columns:
        n_con_vector = df["embeddings_word2vec_avg"].apply(
            lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
        ).sum()

    stats = html.Div([
        stat_card(f"{vocab:,}" if isinstance(vocab, int) else vocab,
                  "Vocabulario", "palabras entrenadas", ACCENT1, "📚"),
        stat_card(str(dim),        "Dimensiones",  "por vector de palabra",  ACCENT2, "📐"),
        stat_card(f"{n_con_vector:,}", "Con vector",  "canciones en MongoDB",   ACCENT3, "🔢"),
        stat_card(models,          "Modelos",      "disponibles",            "#10B981", "🤖"),
    ], className="stat-grid")

    opts = [
        {"label": g, "value": g,
         "label_style": {"color": cmap.get(g, ACCENT1)}}
        for g in generos
    ]
    return stats, opts, generos[:5] if generos else []


@callback(
    Output("w2v-vecinos-chart", "figure"),
    Input("w2v-btn-vecinos", "n_clicks"),
    State("w2v-word-input", "value"),
    State("w2v-model-sel", "value"),
    prevent_initial_call=True,
)
def update_vecinos(_, word, model):
    if not word:
        return empty_fig("Escribe una palabra para buscar vecinos")
    svc = _get_w2v()
    if not svc:
        return empty_fig("Modelo Word2Vec no cargado — ejecuta el notebook de entrenamiento")

    vecinos = svc.most_similar(word.lower().strip(), topn=12, model=model)
    if not vecinos:
        return empty_fig(f"'{word}' no está en el vocabulario")

    words  = [v[0] for v in vecinos][::-1]
    scores = [v[1] for v in vecinos][::-1]

    fig = go.Figure(go.Bar(
        y=words, x=scores,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, _hex_to_rgba(ACCENT2, 0.38)], [1, ACCENT1]],
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(size=10, family=FONT_MONO, color=TEXT_SEC),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=320,
        title=dict(text=f"Vecinos de '{word}' ({model})",
                   font=dict(size=11, color=TEXT_SEC)),
    )
    fig.update_xaxes(title="Similitud coseno", range=[min(scores) * 0.95, 1.05])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


@callback(
    Output("w2v-analogia-result", "children"),
    Input("w2v-btn-analogia", "n_clicks"),
    State("w2v-ana-a", "value"),
    State("w2v-ana-b", "value"),
    State("w2v-ana-c", "value"),
    prevent_initial_call=True,
)
def update_analogia(_, a, b, c):
    if not all([a, b, c]):
        return html.Div("Completa los tres campos.", style={"color": TEXT_SEC, "fontSize": "12px"})
    svc = _get_w2v()
    if not svc:
        return html.Div("Modelo no cargado.", style={"color": "#EF4444"})

    res = svc.analogy(a.strip(), b.strip(), c.strip(), topn=5)
    if not res:
        return html.Div("Alguna palabra no está en el vocabulario.",
                        style={"color": "#EF4444", "fontSize": "12px"})

    formula = html.Div([
        html.Span(b, style={"color": ACCENT1, "fontWeight": "700", "fontFamily": FONT_MONO}),
        html.Span(" − ", style={"color": TEXT_SEC}),
        html.Span(a, style={"color": ACCENT2, "fontWeight": "700", "fontFamily": FONT_MONO}),
        html.Span(" + ", style={"color": TEXT_SEC}),
        html.Span(c, style={"color": ACCENT1, "fontWeight": "700", "fontFamily": FONT_MONO}),
        html.Span(" ≈ ", style={"color": TEXT_SEC, "fontSize": "18px"}),
        html.Span(res[0][0], style={"color": "#10B981", "fontWeight": "800",
                                    "fontSize": "18px", "fontFamily": FONT_MONO}),
    ], style={"fontSize": "14px", "marginBottom": "12px"})

    rows = []
    for i, (word, score) in enumerate(res, 1):
        rows.append(html.Div([
            html.Span(f"#{i}", style={"color": TEXT_MUTED, "fontSize": "11px",
                                      "width": "24px", "fontFamily": FONT_MONO}),
            html.Span(word,  style={"flex": "1", "color": TEXT_PRI, "fontWeight": "600"}),
            html.Span(f"{score:.4f}", style={"color": ACCENT1, "fontSize": "12px",
                                              "fontFamily": FONT_MONO}),
        ], style={"display": "flex", "gap": "10px", "padding": "6px 0",
                  "borderBottom": f"1px solid {BORDER}"}))

    return html.Div([formula, html.Div(rows)])


@callback(
    Output("w2v-genre-sim-chart", "figure"),
    Input("w2v-stats-row", "id"),
)
def update_genre_sim(_):
    svc = _get_w2v()
    df  = get_corpus_df()
    if not svc or df.empty:
        return empty_fig("Modelo no disponible")

    generos = get_generos()
    try:
        sim_df = svc.genre_similarity_matrix(df, col_lyrics="letra", col_genre="genero")
        if sim_df.empty:
            return empty_fig("No se pudo calcular la similitud entre géneros")

        labels = sim_df.index.tolist()
        sim    = sim_df.values

        fig = go.Figure(go.Heatmap(
            z=sim, x=labels, y=labels,
            colorscale=[[0, "#0A0E1A"], [0.5, _hex_to_rgba(ACCENT2, 0.53)], [1, ACCENT1]],
            zmin=0, zmax=1,
            text=np.round(sim, 3),
            texttemplate="%{text:.3f}",
            textfont=dict(size=10),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=360)
        fig.update_xaxes(tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)")
        fig.update_yaxes(tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)")
        return fig
    except Exception as e:
        return empty_fig(f"Error: {e}")


@callback(
    Output("w2v-tsne-chart", "figure"),
    Input("w2v-tsne-genres", "value"),
)
def update_tsne(generos_sel):
    if not generos_sel:
        return empty_fig("Selecciona al menos un género")

    # ← Lee los embeddings directo de MongoDB (vector promedio word2vec)
    emb = _get_doc_embeddings()
    df  = get_corpus_df()

    if emb is None or df.empty:
        return empty_fig(
            "Embeddings word2vec no encontrados en MongoDB. "
            "Ejecuta actualizar_embeddings_mongodb() del notebook de entrenamiento."
        )

    # Alinear: solo filas que tienen vector
    mask_tiene_vec = df["embeddings_word2vec_avg"].apply(
        lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
    )
    df_v = df[mask_tiene_vec].copy().reset_index(drop=True)

    # Re-construir la matriz alineada
    emb_v = np.array(df_v["embeddings_word2vec_avg"].tolist())

    # Filtrar por géneros seleccionados
    mask_genero = df_v["genero"].isin(generos_sel)
    df_f  = df_v[mask_genero].copy()
    emb_f = emb_v[mask_genero.values]

    if df_f.empty:
        return empty_fig("Sin datos para la selección de géneros")

    # Submuestra para t-SNE
    sample_n = min(1500, len(df_f))
    idx = np.random.default_rng(42).choice(len(df_f), sample_n, replace=False)
    df_s   = df_f.iloc[idx]
    emb_s  = emb_f[idx]

    tsne   = TSNE(n_components=2, perplexity=min(30, sample_n - 1),
                  random_state=42, max_iter=500)
    coords = tsne.fit_transform(emb_s)

    cmap = genre_color_map(generos_sel)
    fig  = go.Figure()
    for g in sorted(generos_sel):
        m = df_s["genero"].values == g
        fig.add_trace(go.Scatter(
            x=coords[m, 0], y=coords[m, 1],
            mode="markers", name=g,
            marker=dict(color=cmap.get(g, ACCENT1), size=4, opacity=0.7),
            hovertemplate=f"<b>{g}</b><extra></extra>",
        ))
    fig.update_layout(**PLOTLY_LAYOUT, height=340)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, font=dict(size=10)))
    return fig