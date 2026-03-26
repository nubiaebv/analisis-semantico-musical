"""
dashboard/components.py
=======================
Componentes UI reutilizables y constantes de tema para todos los pages.
"""

from dash import html, dcc
import plotly.graph_objects as go

# ── Paleta de colores ────────────────────────────────────────────────────────
DARK_BG    = "#0A0E1A"
DARK_CARD  = "#111827"
DARK_PANEL = "#1A2235"
BORDER     = "#1E2D45"
ACCENT1    = "#00D4FF"
ACCENT2    = "#7C3AED"
ACCENT3    = "#F59E0B"
SUCCESS    = "#10B981"
DANGER     = "#EF4444"
TEXT_PRI   = "#E2E8F0"
TEXT_SEC   = "#94A3B8"
TEXT_MUTED = "#475569"
FONT_MONO  = "'JetBrains Mono', monospace"

# Colores por género (se completan dinámicamente con los géneros del corpus)
GENRE_PALETTE = [
    "#00D4FF", "#7C3AED", "#F59E0B", "#10B981",
    "#EF4444", "#F472B6", "#34D399", "#FB923C",
    "#A78BFA", "#38BDF8", "#4ADE80", "#FCD34D",
]

# ── Template Plotly base ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRI, family="'Inter', sans-serif", size=11),
    xaxis=dict(gridcolor=BORDER, zeroline=False, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zeroline=False, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
                font=dict(size=11)),
    margin=dict(l=0, r=0, t=8, b=0),
)


def empty_fig(msg: str = "") -> go.Figure:
    """Figura vacía con fondo transparente."""
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT)
    if msg:
        fig.add_annotation(
            text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=TEXT_SEC, size=12),
        )
    return fig


def genre_color_map(generos: list[str]) -> dict[str, str]:
    """Asigna colores a la lista de géneros del corpus."""
    return {g: GENRE_PALETTE[i % len(GENRE_PALETTE)] for i, g in enumerate(sorted(generos))}


# ── Componentes ─────────────────────────────────────────────────────────────

def stat_card(valor, label, subtexto="", color=ACCENT1, icon=""):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "20px", "marginRight": "10px"}) if icon else None,
            html.Div([
                html.Div(str(valor), className="stat-value", style={"color": color}),
                html.Div(label, className="stat-label"),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(subtexto, className="stat-sub") if subtexto else None,
    ], className="stat-card")


def section_header(titulo: str, subtitulo: str = ""):
    return html.Div([
        html.Div(titulo, className="section-title"),
        html.Div(subtitulo, className="section-sub") if subtitulo else None,
        html.Div(className="section-line"),
    ], className="section-header")


def page_header(titulo: str, subtitulo: str = "", badge: str = "", badge_class: str = "badge-sparse"):
    return html.Div([
        html.Div([
            html.H1([
                titulo,
                html.Span(badge, className=f"page-badge {badge_class}") if badge else None,
            ], className="page-title"),
            html.Div(subtitulo, className="page-subtitle") if subtitulo else None,
        ]),
    ], className="page-header")


def info_box(texto: str, amber: bool = False):
    cls = "info-box info-box-amber" if amber else "info-box"
    return html.Div(texto, className=cls)


def card(children, accent: str = "top"):
    cls = {"top": "card card-accent-top", "purple": "card card-accent-purple",
           "amber": "card card-accent-amber"}.get(accent, "card")
    return html.Div(children, className=cls)


def result_row(rank: int, titulo: str, artista: str, genero: str,
               anio, score=None, ocurrencias=None, fragmento=None,
               color_map: dict = None):
    color_map = color_map or {}
    genre_color = color_map.get(genero, ACCENT1)
    is_top = rank <= 3

    meta_parts = [artista]
    if anio:
        meta_parts.append(str(int(anio)) if anio == anio else "")
    meta_str = " · ".join(p for p in meta_parts if p)

    right_widget = None
    if score is not None:
        right_widget = html.Div(f"{score:.3f}", className="result-score")
    elif ocurrencias is not None:
        right_widget = html.Div(
            f"×{int(ocurrencias)}",
            className="result-score",
            style={"background": f"{genre_color}20", "color": genre_color},
        )

    return html.Div([
        html.Div(f"#{rank}", className=f"result-rank {'top' if is_top else ''}"),
        html.Div([
            html.Div(titulo, className="result-title"),
            html.Div([
                meta_str,
                html.Span(
                    genero,
                    className="genre-tag",
                    style={"background": f"{genre_color}20",
                           "color": genre_color,
                           "marginLeft": "8px"},
                ),
            ], className="result-meta", style={"display": "flex", "alignItems": "center"}),
            html.Div(
                html.Em(f'"{fragmento}"', style={"fontSize": "11px", "color": TEXT_MUTED}),
                style={"marginTop": "4px"}
            ) if fragmento else None,
        ], style={"flex": "1"}),
        right_widget,
    ], className="result-row")
