"""
dashboard/pages/busqueda.py
===========================
Página: Búsqueda Semántica Interactiva
Consulta directa a MongoDB Atlas por palabra, género, artista o año.
"""

from dash import html, dcc, Input, Output, State, callback, register_page
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.components import (
    PLOTLY_LAYOUT, ACCENT1, ACCENT2, ACCENT3, BORDER,
    TEXT_PRI, TEXT_SEC, TEXT_MUTED, DARK_CARD, DARK_PANEL, FONT_MONO,
    empty_fig, genre_color_map,
    stat_card, section_header, page_header, info_box, card,
)
from dashboard.db import get_corpus_df, get_generos, buscar_por_palabra

register_page(__name__, path="/busqueda", name="Búsqueda")

# ── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div([
    page_header(
        "Búsqueda Semántica Interactiva",
        "Consulta directa al corpus en MongoDB Atlas — letras, géneros y artistas",
        badge="EN VIVO",
        badge_class="badge-live",
    ),

    info_box(
        "🔍 Escribe cualquier palabra o frase (en inglés) para buscar canciones "
        "cuyas letras la contienen. Los resultados se ordenan por número de ocurrencias. "
        "Puedes filtrar por género y ajustar el número de resultados.",
    ),

    # Panel de búsqueda
    html.Div([
        card([
            section_header("Parámetros de búsqueda"),
            html.Div([
                # Búsqueda principal
                html.Div([
                    html.Label("Palabra o frase", style={"fontSize": "10px",
                               "color": TEXT_SEC, "textTransform": "uppercase",
                               "letterSpacing": "1px", "marginBottom": "6px",
                               "display": "block"}),
                    dcc.Input(
                        id="search-word-input",
                        type="text",
                        placeholder="ej: love, heartbreak, guitar, night…",
                        className="search-input",
                        debounce=False,
                        style={"width": "100%"},
                    ),
                ], style={"flex": "2", "minWidth": "200px"}),

                # Filtro por género
                html.Div([
                    html.Label("Género (opcional)", style={"fontSize": "10px",
                               "color": TEXT_SEC, "textTransform": "uppercase",
                               "letterSpacing": "1px", "marginBottom": "6px",
                               "display": "block"}),
                    dcc.Dropdown(
                        id="search-genre-filter",
                        placeholder="Todos los géneros",
                        clearable=True,
                        style={"background": DARK_CARD, "fontSize": "12px"},
                    ),
                ], style={"flex": "1", "minWidth": "160px"}),

                # Número de resultados
                html.Div([
                    html.Label("Top N resultados", style={"fontSize": "10px",
                               "color": TEXT_SEC, "textTransform": "uppercase",
                               "letterSpacing": "1px", "marginBottom": "6px",
                               "display": "block"}),
                    dcc.Slider(
                        id="search-topn-slider",
                        min=5, max=30, step=5, value=10,
                        marks={5: "5", 10: "10", 15: "15", 20: "20", 30: "30"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "1", "minWidth": "160px"}),

                # Botón
                html.Div([
                    html.Label("\u00a0", style={"display": "block", "marginBottom": "6px"}),
                    html.Button(
                        "🔍  Buscar",
                        id="search-btn",
                        className="btn-primary",
                        style={"width": "100%", "padding": "11px 20px"},
                    ),
                ], style={"flex": "0 0 120px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                      "alignItems": "flex-end"}),
        ], accent="top"),
    ]),

    # Resultados
    html.Div([
        # Lista de canciones
        html.Div([
            card([
                section_header("Canciones encontradas",
                               "Ordenadas por número de ocurrencias de la palabra"),
                html.Div(id="search-results-list",
                         style={"minHeight": "200px"}),
            ], accent="top"),
        ], style={"flex": "1.3", "minWidth": "320px"}),

        # Panel derecho: gráficos
        html.Div([
            card([
                section_header("Ocurrencias por género",
                               "Distribución de la palabra en el corpus"),
                dcc.Graph(id="search-genre-dist",
                          config={"displayModeBar": False}, style={"height": "230px"}),
            ], accent="purple"),

            card([
                section_header("Top artistas",
                               "Artistas con más canciones que contienen la palabra"),
                dcc.Graph(id="search-artist-chart",
                          config={"displayModeBar": False}, style={"height": "230px"}),
            ], accent="amber"),
        ], style={"flex": "1", "minWidth": "280px", "display": "flex",
                  "flexDirection": "column", "gap": "0"}),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),

    # Estadísticas de la búsqueda
    html.Div(id="search-stats-row", className="stat-grid",
             style={"marginTop": "20px"}),

    # Explorador de letras
    html.Div([
        card([
            section_header("Explorador de letras",
                           "Haz clic en una canción de los resultados para ver su letra"),
            html.Div(id="search-lyric-viewer",
                     style={"minHeight": "120px"}),
        ], accent="top"),
    ], id="search-lyric-section", style={"display": "none"}),

    # Store para canción seleccionada
    dcc.Store(id="search-selected-song"),
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("search-genre-filter", "options"),
    Input("search-genre-filter", "id"),
)
def load_genres(_):
    return [{"label": g, "value": g} for g in get_generos()]


@callback(
    Output("search-results-list",  "children"),
    Output("search-genre-dist",    "figure"),
    Output("search-artist-chart",  "figure"),
    Output("search-stats-row",     "children"),
    Output("search-lyric-section", "style"),
    Input("search-btn", "n_clicks"),
    State("search-word-input",    "value"),
    State("search-genre-filter",  "value"),
    State("search-topn-slider",   "value"),
    prevent_initial_call=True,
)
def do_search(_, word, genre_filter, top_n):
    hidden = {"display": "none"}
    empty_list = html.Div(
        "Ingresa una palabra y presiona Buscar.",
        style={"color": TEXT_SEC, "fontSize": "12px", "padding": "20px 0"},
    )

    if not word or not word.strip():
        return empty_list, empty_fig(), empty_fig(), html.Div(), hidden

    word = word.strip()
    df_r = buscar_por_palabra(word, top_n=top_n, genero=genre_filter)
    generos_full = get_generos()
    cmap = genre_color_map(generos_full)

    if df_r.empty:
        no_res = html.Div(
            [html.Div(f"No se encontraron canciones con '{word}'.",
                      style={"color": TEXT_SEC, "fontSize": "13px"}),
             html.Div("Intenta con otra palabra o amplía el filtro de género.",
                      style={"color": TEXT_MUTED, "fontSize": "11px", "marginTop": "4px"})],
            style={"padding": "20px 0"}
        )
        return no_res, empty_fig("Sin resultados"), empty_fig("Sin resultados"), html.Div(), hidden

    # ── Lista de resultados ──────────────────────────────────────────────────
    rows = []
    for rank, (_, row) in enumerate(df_r.iterrows(), 1):
        gc     = cmap.get(str(row.get("genero", "")), ACCENT1)
        titulo  = str(row.get("titulo", "—"))
        artista = str(row.get("artista", "—"))
        genero  = str(row.get("genero", "—"))
        anio    = row.get("anio", "")
        ocur    = int(row.get("ocurrencias", 0))
        frag    = str(row.get("fragmento", ""))

        # Resaltar la palabra en el fragmento
        frag_hl = frag.replace(
            word,
            f"【{word}】"  # marcador visual simple
        ).replace(
            word.lower(),
            f"【{word}】"
        )

        rows.append(html.Div([
            html.Div(f"#{rank}", className=f"result-rank {'top' if rank <= 3 else ''}"),
            html.Div([
                html.Div(titulo, className="result-title"),
                html.Div([
                    artista,
                    html.Span(str(int(anio)) if anio and anio == anio else "",
                              style={"color": TEXT_MUTED, "marginLeft": "6px"}),
                    html.Span(genero, className="genre-tag",
                              style={"background": f"{gc}20", "color": gc,
                                     "marginLeft": "8px"}),
                ], className="result-meta",
                   style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
                html.Div(
                    html.Em(f'…{frag_hl[:160]}…',
                            style={"fontSize": "11px", "color": TEXT_MUTED,
                                   "lineHeight": "1.5"}),
                    style={"marginTop": "4px"},
                ) if frag else None,
            ], style={"flex": "1"}),
            html.Div([
                html.Div(f"×{ocur}", style={
                    "color": gc, "fontWeight": "800", "fontFamily": FONT_MONO,
                    "fontSize": "13px", "background": f"{gc}15",
                    "padding": "4px 10px", "borderRadius": "4px",
                    "textAlign": "center",
                }),
                html.Div("ocurrencias", style={"fontSize": "9px", "color": TEXT_MUTED,
                                                "textAlign": "center", "marginTop": "2px"}),
            ]),
        ], className="result-row"))

    results_div = html.Div(rows)

    # ── Gráfico distribución por género ─────────────────────────────────────
    if "genero" in df_r.columns:
        genre_counts = df_r.groupby("genero")["ocurrencias"].sum().sort_values(ascending=True)
        colors_bar   = [cmap.get(g, ACCENT1) for g in genre_counts.index]

        fig_genres = go.Figure(go.Bar(
            y=genre_counts.index.tolist(),
            x=genre_counts.values.tolist(),
            orientation="h",
            marker_color=colors_bar,
            text=genre_counts.values.tolist(),
            textposition="outside",
            textfont=dict(size=10, family=FONT_MONO),
        ))
        fig_genres.update_layout(**PLOTLY_LAYOUT, height=230,
                                  xaxis=dict(title=f'Ocurrencias de "{word}"'),
                                  yaxis=dict(gridcolor="transparent"))
    else:
        fig_genres = empty_fig()

    # ── Gráfico top artistas ──────────────────────────────────────────────
    if "artista" in df_r.columns:
        artist_counts = (df_r.groupby("artista")["ocurrencias"]
                         .sum()
                         .sort_values(ascending=False)
                         .head(8)
                         .sort_values(ascending=True))
        fig_artists = go.Figure(go.Bar(
            y=artist_counts.index.tolist(),
            x=artist_counts.values.tolist(),
            orientation="h",
            marker=dict(
                color=artist_counts.values.tolist(),
                colorscale=[[0, f"{ACCENT2}60"], [1, ACCENT1]],
            ),
            text=artist_counts.values.tolist(),
            textposition="outside",
            textfont=dict(size=10, family=FONT_MONO),
        ))
        fig_artists.update_layout(**PLOTLY_LAYOUT, height=230,
                                   xaxis=dict(title="Ocurrencias totales"),
                                   yaxis=dict(gridcolor="transparent"))
    else:
        fig_artists = empty_fig()

    # ── Stats de la búsqueda ─────────────────────────────────────────────
    total_ocur    = int(df_r["ocurrencias"].sum()) if "ocurrencias" in df_r.columns else 0
    generos_found = df_r["genero"].nunique() if "genero" in df_r.columns else 0
    artistas_fnd  = df_r["artista"].nunique() if "artista" in df_r.columns else 0

    stats_row = html.Div([
        stat_card(f"{len(df_r)}", "Canciones", f"con '{word}' en la letra", ACCENT1, "🎵"),
        stat_card(f"{total_ocur:,}", "Ocurrencias", "total en el corpus filtrado", ACCENT2, "🔢"),
        stat_card(f"{artistas_fnd}", "Artistas", "distintos", ACCENT3, "🎤"),
        stat_card(f"{generos_found}", "Géneros", "representados", "#10B981", "🎸"),
    ], className="stat-grid")

    return results_div, fig_genres, fig_artists, stats_row, {"display": "block"}
