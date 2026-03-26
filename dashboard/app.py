"""
dashboard/app.py
================
Punto de entrada del Dashboard — Análisis Semántico Musical
Proyecto 2 · Minería de Textos · CUC

Estructura:
    dashboard/
    ├── app.py              ← este archivo (entry point)
    ├── db.py               ← conexión MongoDB Atlas
    ├── components.py       ← UI reutilizables y tema
    ├── assets/
    │   └── style.css       ← estilos globales
    └── pages/
        ├── bow_tfidf.py    ← BoW / TF-IDF
        ├── word2vec.py     ← Word2Vec
        ├── beto.py         ← BERT (bert-base-uncased)
        ├── comparacion.py  ← Comparación final
        └── busqueda.py     ← Búsqueda interactiva MongoDB

Uso:
    cd <PROJECT_ROOT>
    python dashboard/app.py
    # Abre http://127.0.0.1:8050
"""

import sys
import os
from pathlib import Path

# ── Garantizar que PROJECT_ROOT esté en sys.path ─────────────────────────────
# Necesario para que todos los módulos del proyecto sean importables
_this_dir = Path(__file__).resolve().parent          # dashboard/
_root     = _this_dir.parent                          # PROJECT_ROOT (contiene src/)
for _p in [str(_root), str(_root / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports Dash ─────────────────────────────────────────────────────────────
import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

from dashboard.components import (
    DARK_BG, DARK_CARD, DARK_PANEL, BORDER, ACCENT1, ACCENT2,
    TEXT_PRI, TEXT_SEC, TEXT_MUTED, FONT_MONO,
)
from dashboard.db import get_stats, get_generos

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(_this_dir / "pages"),
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="AnálisisMusicAI · Proyecto 2 CUC",
)
server = app.server  # Para despliegue con Gunicorn/Waitress

# ── Navegación ────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("🏠", "Inicio",             "/"),
    ("📦", "BoW / TF-IDF",       "/bow-tfidf"),
    ("⚡", "Word2Vec",           "/word2vec"),
    ("🧠", "BERT",               "/beto"),
    ("📊", "Comparación Final",  "/comparacion"),
    ("🔍", "Búsqueda",           "/busqueda"),
]

def nav_button(icon, label, href):
    return html.A(
        html.Div([
            html.Span(icon, className="nav-btn-icon"),
            html.Span(label),
        ], className="nav-btn", id=f"nav-{href.strip('/') or 'home'}"),
        href=href,
        style={"textDecoration": "none"},
    )

sidebar = html.Div([
    # Logo
    html.Div([
        html.Div("♪", className="sidebar-logo-icon"),
        html.Div([
            html.Div("AnálisisMusicAI", className="sidebar-logo-text"),
            html.Div("Proyecto 2 · CUC", className="sidebar-logo-sub"),
        ]),
    ], className="sidebar-logo"),

    # Nav links
    html.Div("NAVEGACIÓN", className="sidebar-section-label"),
    html.Div([nav_button(icon, label, href) for icon, label, href in NAV_ITEMS]),

    # Footer
    html.Div([
        html.Div("Minería de Textos"),
        html.Div("Prof. Osvaldo González", style={"marginTop": "2px"}),
    ], className="sidebar-footer"),

], className="sidebar")

# ── Layout raíz ──────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="_url", refresh=False),
    sidebar,
    html.Div(
        dash.page_container,
        className="main-content",
    ),
], className="app-wrapper")


# ── Página de inicio (index) ──────────────────────────────────────────────────
dash.register_page(
    "home",
    path="/",
    name="Inicio",
    layout=html.Div(id="home-layout"),
)

@callback(Output("home-layout", "children"), Input("home-layout", "id"))
def render_home(_):
    stats = get_stats()
    generos = get_generos()

    total    = stats.get("total", "—")
    n_genres = stats.get("generos", "—")
    artists  = stats.get("artistas", "—")
    yr_range = f"{stats.get('anio_min','?')} – {stats.get('anio_max','?')}"
    avg_w    = stats.get("avg_palabras", "—")

    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1("Dashboard de Análisis Semántico Musical",
                        style={"fontSize": "24px", "fontWeight": "800",
                               "color": TEXT_PRI, "margin": "0 0 6px 0"}),
                html.Div("Proyecto 2 · Minería de Textos · CUC · Corpus en inglés — MongoDB Atlas",
                         style={"color": TEXT_SEC, "fontSize": "13px"}),
            ]),
            html.Div("♪", style={"fontSize": "48px", "color": ACCENT1, "opacity": "0.7"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "28px",
                  "paddingBottom": "20px", "borderBottom": f"1px solid {BORDER}"}),

        # Stats del corpus
        html.Div([
            _home_stat(f"{total:,}" if isinstance(total, int) else total,
                       "Canciones", "en MongoDB Atlas", ACCENT1, "🎵"),
            _home_stat(str(n_genres), "Géneros", "categorías musicales", ACCENT2, "🎸"),
            _home_stat(f"{artists:,}" if isinstance(artists, int) else artists,
                       "Artistas", "en el corpus", ACCENT3, "🎤"),
            _home_stat(yr_range, "Período", "años cubiertos", "#10B981", "📅"),
            _home_stat(f"{int(avg_w):,}" if isinstance(avg_w, float) else avg_w,
                       "Palabras promedio", "por canción", "#F472B6", "📝"),
        ], className="stat-grid", style={"marginBottom": "28px"}),

        # Cards de navegación
        html.Div("MÓDULOS DE ANÁLISIS",
                 style={"fontSize": "10px", "color": TEXT_MUTED, "letterSpacing": "2px",
                        "marginBottom": "16px"}),
        html.Div([
            _nav_card("📦", "BoW / TF-IDF", "/bow-tfidf", ACCENT1,
                      "Representación dispersa. Línea base del proyecto.",
                      ["Matrices dispersas", "Top-N palabras por género",
                       "Heatmap de similitud", "Demo de ortogonalidad"]),
            _nav_card("⚡", "Word2Vec", "/word2vec", ACCENT2,
                      "Embeddings estáticos CBOW y Skip-Gram.",
                      ["Vecinos semánticos", "Analogías vectoriales",
                       "Similitud entre géneros", "Visualización t-SNE"]),
            _nav_card("🧠", "BERT", "/beto", ACCENT3,
                      "Embeddings contextuales bert-base-uncased.",
                      ["Polisemia contextual", "Masked Language Model",
                       "Búsqueda semántica", "t-SNE contextual"]),
            _nav_card("📊", "Comparación", "/comparacion", "#10B981",
                      "BoW vs Word2Vec vs BERT — análisis cuantitativo.",
                      ["Clasificación de género", "Clustering K-Means",
                       "Silhouette Score", "t-SNE 3-en-1"]),
            _nav_card("🔍", "Búsqueda", "/busqueda", "#F472B6",
                      "Búsqueda interactiva en MongoDB Atlas.",
                      ["Busca por palabra o frase", "Filtro por género",
                       "Fragmentos con contexto", "Distribución por artista"]),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(auto-fit, minmax(230px, 1fr))",
                  "gap": "16px"}),
    ])


def _home_stat(valor, label, sub, color, icon):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "20px", "marginRight": "10px"}),
            html.Div([
                html.Div(str(valor), style={"fontSize": "22px", "fontWeight": "800",
                                            "color": color, "fontFamily": FONT_MONO,
                                            "lineHeight": "1"}),
                html.Div(label, style={"fontSize": "10px", "color": TEXT_SEC,
                                       "textTransform": "uppercase", "letterSpacing": "1px",
                                       "marginTop": "3px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(sub, style={"fontSize": "11px", "color": TEXT_MUTED, "marginTop": "6px"}),
    ], className="stat-card", style={"borderTop": f"3px solid {color}"})


def _nav_card(icon, title, href, color, desc, features):
    return html.A(
        html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "22px"}),
                html.Div([
                    html.Div(title, style={"color": color, "fontWeight": "800",
                                           "fontSize": "14px"}),
                    html.Div(desc, style={"color": TEXT_SEC, "fontSize": "11px",
                                          "marginTop": "2px"}),
                ]),
            ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start",
                      "marginBottom": "12px"}),
            html.Ul([
                html.Li(f, style={"fontSize": "11px", "color": TEXT_MUTED,
                                   "marginBottom": "3px"})
                for f in features
            ], style={"paddingLeft": "16px", "margin": "0"}),
        ], style={
            "background": DARK_PANEL,
            "border": f"1px solid {BORDER}",
            "borderTop": f"3px solid {color}",
            "borderRadius": "10px",
            "padding": "18px",
            "cursor": "pointer",
            "transition": "border-color 0.2s ease",
        }),
        href=href,
        style={"textDecoration": "none"},
    )


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050, host="127.0.0.1")
