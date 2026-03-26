"""
dashboard/pages/beto.py
=======================
Página: Análisis BERT (bert-base-uncased)

Cambios respecto a la versión original:
  - BetoService ahora se importa desde src.services.embeddings_beto
    (adaptador que usa el backend real CargadorBETO / embedding_cls / AnalizadorMLM)
  - _get_bert_embeddings() lee los vectores beto_cls directamente de MongoDB
    (columna embeddings_beto_cls del DataFrame de consultar_base_datos)
    en lugar de cargar un archivo .npy externo.
  - Se corrige el bug: antes llamaba a embeddings_word2vec() en lugar
    de embeddings_beto().
"""

import sys
import numpy as np
from pathlib import Path
from dash import html, dcc, Input, Output, State, callback, register_page
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from dashboard.components import (
    PLOTLY_LAYOUT, ACCENT1, ACCENT2, ACCENT3, BORDER,
    TEXT_PRI, TEXT_SEC, TEXT_MUTED, DARK_CARD, FONT_MONO,
    empty_fig, genre_color_map,
    stat_card, section_header, page_header, info_box, card,
)
from dashboard.db import get_corpus_df, get_generos, PROJECT_ROOT

register_page(__name__, path="/beto", name="BERT")

# ── Helper: hex (#RRGGBB) → rgba(r,g,b,alpha) ────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Cache ────────────────────────────────────────────────────────────────────
_bert_cache: dict = {}


def _get_bert_embeddings() -> np.ndarray:
    """
    Lee los embeddings BERT CLS directamente de MongoDB.
    Retorna np.ndarray (n, 768) o None si no hay vectores.
    Bugfix: antes llamaba a embeddings_word2vec() — ahora usa embeddings_beto_cls.
    """
    df = get_corpus_df()
    if df.empty or "embeddings_beto_cls" not in df.columns:
        return None

    mask = df["embeddings_beto_cls"].apply(
        lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
    )
    if not mask.any():
        return None

    try:
        return np.array(df.loc[mask, "embeddings_beto_cls"].tolist())
    except Exception:
        return None


def _get_bert_service():
    """
    Carga BetoService en memoria (singleton en _bert_cache).
    Usa el adaptador src/services/embeddings_beto.py que delega
    al backend real (CargadorBETO, AnalizadorMLM, etc.).
    """
    if "svc" in _bert_cache:
        return _bert_cache["svc"]

    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        # ← nuevo import: adaptador que usa el backend real
        from src.services.embeddings_beto import BetoService
        svc = BetoService(model_name="bert-base-uncased", batch_size=16, max_length=128)
        svc.load_model()
        _bert_cache["svc"] = svc
        return svc
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("No se pudo cargar BetoService: %s", e)
        _bert_cache["svc"] = None
        return None


# ── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div([
    page_header(
        "BERT — bert-base-uncased",
        "Embeddings contextuales — la misma palabra, distintos vectores según contexto",
        badge="CONTEXTUAL",
        badge_class="badge-context",
    ),

    html.Div(id="bert-stats-row", className="stat-grid"),

    # Fila 1: Polisemia + MLM
    html.Div([
        html.Div([
            card([
                section_header("Polisemia contextual",
                               "BERT asigna vectores distintos a la misma palabra según su contexto"),
                info_box("💡 Word2Vec asigna UN vector a 'rock' en todos los contextos. "
                         "BERT asigna vectores distintos: rock (música) vs rock (piedra)."),
                html.Div([
                    dcc.Input(id="bert-poly-word", type="text",
                              placeholder="Palabra polisémica… ej: rock, band, beat",
                              className="search-input", style={"marginBottom": "8px"}),
                    html.Div([
                        dcc.Textarea(
                            id="bert-poly-ctx1",
                            placeholder="Contexto 1… ej: I love rock music and loud guitars.",
                            className="search-input",
                            style={"height": "60px", "resize": "none"},
                        ),
                        dcc.Textarea(
                            id="bert-poly-ctx2",
                            placeholder="Contexto 2… ej: She sat on a rock by the river.",
                            className="search-input",
                            style={"height": "60px", "resize": "none"},
                        ),
                        dcc.Textarea(
                            id="bert-poly-ctx3",
                            placeholder="Contexto 3… ej: The mother rocked her baby to sleep.",
                            className="search-input",
                            style={"height": "60px", "resize": "none"},
                        ),
                    ], style={"display": "flex", "flexDirection": "column", "gap": "6px",
                              "marginBottom": "10px"}),
                    html.Button("Calcular polisemia", id="bert-btn-poly",
                                className="btn-primary"),
                ]),
                html.Div(id="bert-poly-result", style={"marginTop": "14px"}),
            ], accent="amber"),
        ], style={"flex": "1", "minWidth": "300px"}),

        html.Div([
            card([
                section_header("Masked Language Model",
                               "BERT predice la palabra enmascarada [MASK] según el contexto"),
                info_box("Escribe una oración con [MASK] para ver qué predice BERT. "
                         "Útil para explorar asociaciones por género musical."),
                dcc.Textarea(
                    id="bert-mlm-input",
                    placeholder='Ej: Hip hop always talks about [MASK] and street life.',
                    className="search-input",
                    style={"height": "80px", "resize": "none", "marginBottom": "10px"},
                ),
                html.Button("Predecir [MASK]", id="bert-btn-mlm",
                            className="btn-primary", style={"marginBottom": "14px"}),
                html.Div(id="bert-mlm-result"),

                html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),
                section_header("Plantillas por género",
                               "Predefinidas para explorar diferencias semánticas"),
                html.Div([
                    html.Button(
                        template[:45] + "…",
                        id={"type": "bert-template-btn", "index": i},
                        className="btn-secondary",
                        style={"marginBottom": "6px", "width": "100%",
                               "textAlign": "left", "fontSize": "11px"},
                    )
                    for i, template in enumerate([
                        "Hip hop always talks about [MASK] and street life.",
                        "Pop music is full of [MASK] and dancing.",
                        "Rock music expresses [MASK] and rebellion.",
                        "R&B songs are about [MASK] and deep emotions.",
                        "The singer felt [MASK] after the concert.",
                    ])
                ]),
                dcc.Store(id="bert-templates-store", data=[
                    "Hip hop always talks about [MASK] and street life.",
                    "Pop music is full of [MASK] and dancing.",
                    "Rock music expresses [MASK] and rebellion.",
                    "R&B songs are about [MASK] and deep emotions.",
                    "The singer felt [MASK] after the concert.",
                ]),
            ], accent="purple"),
        ], style={"flex": "1", "minWidth": "300px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),

    # Fila 2: Búsqueda semántica + t-SNE
    html.Div([
        html.Div([
            card([
                section_header("Búsqueda semántica BERT",
                               "Las 5 canciones más similares a tu consulta (similitud coseno)"),
                dcc.Input(
                    id="bert-search-input",
                    type="text",
                    placeholder="Consulta libre en inglés… ej: heartbreak and lost love",
                    className="search-input",
                    style={"marginBottom": "10px"},
                ),
                html.Button("🔍 Buscar con BERT", id="bert-btn-search",
                            className="btn-primary", style={"marginBottom": "14px"}),
                html.Div(id="bert-search-results"),
            ], accent="top"),
        ], style={"flex": "1", "minWidth": "300px"}),

        html.Div([
            card([
                section_header("t-SNE — Embeddings BERT",
                               "Proyección 2D del espacio semántico contextual (desde MongoDB)"),
                html.Div([
                    dcc.Checklist(
                        id="bert-tsne-genres",
                        inline=True,
                        labelStyle={"marginRight": "12px", "fontSize": "11px",
                                    "color": TEXT_SEC, "cursor": "pointer"},
                        style={"marginBottom": "12px"},
                    ),
                ]),
                dcc.Graph(id="bert-tsne-chart",
                          config={"displayModeBar": False}, style={"height": "350px"}),
            ], accent="top"),
        ], style={"flex": "1.2", "minWidth": "320px"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginTop": "0"}),
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("bert-stats-row", "children"),
    Output("bert-tsne-genres", "options"),
    Output("bert-tsne-genres", "value"),
    Input("bert-stats-row", "id"),
)
def load_bert_stats(_):
    generos = get_generos()
    cmap    = genre_color_map(generos)

    # Contar canciones con vector BERT en MongoDB
    df = get_corpus_df()
    n_con_vector = 0
    if not df.empty and "embeddings_beto_cls" in df.columns:
        n_con_vector = df["embeddings_beto_cls"].apply(
            lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
        ).sum()

    stats = html.Div([
        stat_card(f"{n_con_vector:,}", "Con embedding", "canciones en MongoDB", ACCENT1, "🧠"),
        stat_card("768",               "Dimensiones",   "vector CLS de BERT",   ACCENT2, "📐"),
        stat_card(f"{len(generos)}",   "Géneros",       "en el corpus",          ACCENT3, "🎸"),
        stat_card("bert-base-uncased", "Modelo",        "HuggingFace",           "#10B981", "🤗"),
    ], className="stat-grid")

    opts = [
        {"label": g, "value": g, "label_style": {"color": cmap.get(g, ACCENT1)}}
        for g in generos
    ]
    return stats, opts, generos[:5] if generos else []


@callback(
    Output("bert-poly-result", "children"),
    Input("bert-btn-poly", "n_clicks"),
    State("bert-poly-word", "value"),
    State("bert-poly-ctx1", "value"),
    State("bert-poly-ctx2", "value"),
    State("bert-poly-ctx3", "value"),
    prevent_initial_call=True,
)
def calc_polisemia(_, word, ctx1, ctx2, ctx3):
    contexts = [c for c in [ctx1, ctx2, ctx3] if c and c.strip()]
    if not word or len(contexts) < 2:
        return html.Div("Necesitas la palabra y al menos 2 contextos.",
                        style={"color": TEXT_SEC, "fontSize": "12px"})

    svc = _get_bert_service()
    if not svc:
        return html.Div("BERT no disponible — instala transformers y torch.",
                        style={"color": "#EF4444", "fontSize": "12px"})

    results = svc.polysemy_demo(word.strip(), contexts)
    if not results:
        return html.Div("No se pudo calcular la polisemia.",
                        style={"color": "#EF4444", "fontSize": "12px"})

    rows = []
    for r in results:
        sim   = r["sim"]
        color = ACCENT1 if sim > 0.85 else (ACCENT3 if sim > 0.70 else "#EF4444")
        rows.append(html.Div([
            html.Div([
                html.Div(f'"{r["ctx_a"][:50]}…"',
                         style={"fontSize": "11px", "color": TEXT_SEC}),
                html.Div("vs", style={"color": TEXT_MUTED, "fontSize": "10px",
                                      "margin": "2px 0"}),
                html.Div(f'"{r["ctx_b"][:50]}…"',
                         style={"fontSize": "11px", "color": TEXT_SEC}),
            ], style={"flex": "1"}),
            html.Div(f"{sim:.4f}", style={
                "color": color, "fontWeight": "800", "fontFamily": FONT_MONO,
                "fontSize": "14px", "background": f"{color}15",
                "padding": "4px 10px", "borderRadius": "4px",
            }),
        ], style={"display": "flex", "gap": "12px", "alignItems": "center",
                  "padding": "8px 0", "borderBottom": f"1px solid {BORDER}"}))

    note = "Similitudes más bajas = contextos más distintos = mejor captura de polisemia"
    return html.Div([
        html.Div(note, style={"fontSize": "10px", "color": TEXT_MUTED, "marginBottom": "8px"}),
        html.Div(rows),
    ])


@callback(
    Output("bert-mlm-input", "value"),
    Input({"type": "bert-template-btn", "index": 0}, "n_clicks"),
    Input({"type": "bert-template-btn", "index": 1}, "n_clicks"),
    Input({"type": "bert-template-btn", "index": 2}, "n_clicks"),
    Input({"type": "bert-template-btn", "index": 3}, "n_clicks"),
    Input({"type": "bert-template-btn", "index": 4}, "n_clicks"),
    State("bert-templates-store", "data"),
    prevent_initial_call=True,
)
def fill_template(*args):
    from dash import ctx
    templates = args[-1]
    if not ctx.triggered_id:
        return ""
    idx = ctx.triggered_id.get("index", 0)
    return templates[idx]


@callback(
    Output("bert-mlm-result", "children"),
    Input("bert-btn-mlm", "n_clicks"),
    State("bert-mlm-input", "value"),
    prevent_initial_call=True,
)
def predict_mask(_, sentence):
    if not sentence or "[MASK]" not in sentence:
        return html.Div("La oración debe contener [MASK].",
                        style={"color": TEXT_SEC, "fontSize": "12px"})
    svc = _get_bert_service()
    if not svc:
        return html.Div("BERT no disponible.", style={"color": "#EF4444"})

    preds = svc.fill_mask(sentence, top_k=6)
    if not preds:
        return html.Div("Error al predecir.", style={"color": "#EF4444"})

    max_score = max(p["score"] for p in preds)
    bars = []
    for p in preds:
        pct   = p["score"] / max_score
        color = ACCENT1 if pct > 0.8 else (ACCENT2 if pct > 0.5 else TEXT_MUTED)
        bars.append(html.Div([
            html.Span(p["token"], style={
                "color": color, "fontWeight": "700", "fontFamily": FONT_MONO,
                "fontSize": "12px", "width": "100px", "display": "inline-block",
            }),
            html.Div(style={
                "height": "8px", "borderRadius": "4px",
                "background": f"linear-gradient(90deg, {color}, {color}88)",
                "width": f"{pct * 160}px", "display": "inline-block",
                "verticalAlign": "middle", "margin": "0 8px",
            }),
            html.Span(f"{p['score']:.4f}", style={
                "color": TEXT_SEC, "fontSize": "10px", "fontFamily": FONT_MONO,
            }),
        ], style={"padding": "4px 0"}))
    return html.Div(bars)


@callback(
    Output("bert-search-results", "children"),
    Input("bert-btn-search", "n_clicks"),
    State("bert-search-input", "value"),
    prevent_initial_call=True,
)
def bert_search(_, query):
    if not query:
        return html.Div("Escribe una consulta.", style={"color": TEXT_SEC})

    # ← Lee embeddings de MongoDB (beto_cls), no de .npy
    emb = _get_bert_embeddings()
    df  = get_corpus_df()

    if emb is None or df.empty:
        return html.Div(
            "Embeddings BERT no disponibles en MongoDB. "
            "Ejecuta actualizar_beto_cls_mongodb() del notebook correspondiente.",
            style={"color": "#EF4444", "fontSize": "12px"},
        )

    svc = _get_bert_service()
    if not svc:
        return html.Div("BERT no disponible.", style={"color": "#EF4444"})

    # Alinear DataFrame con la matriz de embeddings
    mask = df["embeddings_beto_cls"].apply(
        lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
    )
    df_aligned = df[mask].reset_index(drop=True)

    results = svc.semantic_search(
        query, emb, df_aligned,
        top_n=5,
        col_title="titulo", col_artist="artista", col_genre="genero",
    )
    cmap = genre_color_map(get_generos())

    rows = []
    for r in results:
        gc = cmap.get(r["genre"], ACCENT1)
        rows.append(html.Div([
            html.Div(f"#{r['rank']}", className=f"result-rank {'top' if r['rank'] <= 3 else ''}"),
            html.Div([
                html.Div(r["title"], className="result-title"),
                html.Div([
                    r["artist"],
                    html.Span(r["genre"], className="genre-tag",
                              style={"background": f"{gc}20", "color": gc,
                                     "marginLeft": "8px"}),
                ], className="result-meta",
                   style={"display": "flex", "alignItems": "center"}),
            ], style={"flex": "1"}),
            html.Div(f"{r['score']:.4f}", className="result-score"),
        ], className="result-row"))

    return html.Div(rows) if rows else html.Div(
        "Sin resultados para esta consulta.", style={"color": TEXT_SEC}
    )


@callback(
    Output("bert-tsne-chart", "figure"),
    Input("bert-tsne-genres", "value"),
)
def update_bert_tsne(generos_sel):
    if not generos_sel:
        return empty_fig("Selecciona al menos un género")

    # ← Lee embeddings de MongoDB
    emb = _get_bert_embeddings()
    df  = get_corpus_df()

    if emb is None or df.empty:
        return empty_fig(
            "Embeddings BERT no encontrados en MongoDB. "
            "Ejecuta actualizar_beto_cls_mongodb()."
        )

    # Alinear: solo filas con vector BERT válido
    mask_vec = df["embeddings_beto_cls"].apply(
        lambda v: isinstance(v, (list, np.ndarray)) and len(v) > 0
    )
    df_v  = df[mask_vec].copy().reset_index(drop=True)
    emb_v = np.array(df_v["embeddings_beto_cls"].tolist())

    # Filtrar por géneros
    mask_g = df_v["genero"].isin(generos_sel)
    df_f   = df_v[mask_g].copy()
    emb_f  = emb_v[mask_g.values]

    if df_f.empty:
        return empty_fig("Sin datos para la selección de géneros")

    sample_n = min(1000, len(df_f))
    idx    = np.random.default_rng(42).choice(len(df_f), sample_n, replace=False)
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
            x=coords[m, 0], y=coords[m, 1], mode="markers", name=g,
            marker=dict(color=cmap.get(g, ACCENT1), size=4, opacity=0.7),
        ))
    fig.update_layout(**PLOTLY_LAYOUT, height=350)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, font=dict(size=10)))
    return fig