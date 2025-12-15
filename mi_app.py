import streamlit as st
import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def load_logo(url, zoom=0.08):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return OffsetImage(img, zoom=zoom)


# =========================
# FUNCIONES AUXILIARES
# =========================
def efficient_frontier(df):
    """
    Devuelve los puntos no dominados (Pareto óptimos)
    maximizando win_rate y return
    """
    pareto_mask = []

    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (
                other["win_rate"] >= row["win_rate"]
                and other["return"] >= row["return"]
                and (
                    other["win_rate"] > row["win_rate"]
                    or other["return"] > row["return"]
                )
            ):
                dominated = True
                break
        pareto_mask.append(not dominated)

    frontier = df[pareto_mask].copy()

    frontier = pd.concat(
        [
            frontier,
            df.loc[[df["win_rate"].idxmax()]],
            df.loc[[df["return"].idxmax()]],
        ]
    ).drop_duplicates()

    frontier = frontier.sort_values("win_rate")
    return frontier


df = pd.read_csv("club_data.csv")
league_map = {
    "ES1": "LaLiga",
    "FR1": "Ligue 1",
    "GB1": "Premier League",
    "L1":  "Bundesliga",
    "IT1": "Serie A"
}
df["league_name"] = df["domestic_competition_id"].map(
    lambda x: f"{league_map.get(x, 'Unknown')} ({x})"
)
# Posición original de la columna
pos = df.columns.get_loc("domestic_competition_id")

# Reordenar columnas
cols = df.columns.tolist()
cols.insert(pos, cols.pop(cols.index("league_name")))
df = df[cols]
df = df.drop(columns=["domestic_competition_id"])


# ==========================
# DATOS BASE
# ==========================
ligas = sorted(df["league_name"].dropna().unique().tolist())
todos_los_clubes = sorted(df["club_name"].dropna().unique().tolist())

st.title("Comparador de clubes")

col_izq, col_der = st.columns(2)

# ==========================
# COLUMNA IZQUIERDA
# ==========================
with col_izq:
    st.subheader("Club a comparar")

    liga_target = st.selectbox(
        "Liga del target club:",
        ["(Todas)"] + ligas
    )

    if liga_target == "(Todas)":
        clubs_target = todos_los_clubes
    else:
        clubs_target = sorted(
            df[df["league_name"] == liga_target]["club_name"]
            .dropna()
            .unique()
            .tolist()
        )

    target_club = st.selectbox(
        "Club objetivo:",
        clubs_target
    )
    club_target_df = df[df["club_name"] == target_club]

# ==========================
# COLUMNA DERECHA
# ==========================
with col_der:
    st.subheader("Comparación")

    ligas_comparar = st.multiselect(
        "Ligas a comparar:",
        ligas
    )

    if ligas_comparar:
        df_comparar = df[df["league_name"].isin(ligas_comparar)]
    else:
        df_comparar = df.copy()

    # --------------------------
    # FILTROS
    # --------------------------
    st.subheader("Filtros")

    # Profitability -> columna "return"
    profitability = st.slider(
        "Profitability (Return)",
        float(df["return"].min()),
        float(df["return"].max()),
        (
            float(df["return"].min()),
            float(df["return"].max())
        )
    )

    # WinRate -> columna "win_rate"
    winrate = st.slider(
        "Win Rate (%)",
        float(df["win_rate"].min()),
        float(df["win_rate"].max()),
        (
            float(df["win_rate"].min()),
            float(df["win_rate"].max())
        )
    )

    # Filtro opcional: solo clubs rentables
    solo_rentables = st.checkbox("Solo clubs rentables")

    df_filtrado = df_comparar[
        (df_comparar["return"].between(*profitability)) &
        (df_comparar["win_rate"].between(*winrate))
    ]

    if solo_rentables:
        df_filtrado = df_filtrado[df_filtrado["is_profitable"] == True]


# ==========================
# RESULTADOS
# ==========================
if df_filtrado.empty:
    st.warning("No se encontraron clubes que cumplan los criterios seleccionados.")
import mplcursors

st.divider()
st.subheader("Resultados")

st.write(f"**Club a comparar:** {target_club}")
# hacer que saga la liga sin [' y ']
ligas_comparar_str = ", ".join(ligas_comparar) if ligas_comparar else "Todas"
st.write(f"**Ligas a comparar:** {ligas_comparar_str}")
st.write(f"**Número de clubes tras filtros:** {len(df_filtrado)}")
df = df_filtrado.copy()
# ==========================
# PARETO FRONTIER PLOT
# ==========================

if not club_target_df.empty:
    # Comprobar si ya está en df_filtrado
    if target_club not in df_filtrado["club_name"].values:
        # Añadirlo al DataFrame filtrado
        df = pd.concat([df, club_target_df], ignore_index=True)
else:
    st.warning(f"No se encontró el club objetivo: {target_club}")



# BASE FIGURE (GRAY POINTS)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["win_rate"],
    y=df["return"],
    mode="markers",
    marker=dict(size=10, color="gray", opacity=0.55),
    text=df["club_name"],
    hovertemplate="<b>%{text}</b><br>Win Rate: %{x:.1f}%<br>Return: €%{y:,.0f}<extra></extra>",
    name="All Clubs"
))

# REFERENCE LINES
fig.add_hline(y=0, line_color='black', line_width=1, line_dash='dash')
fig.add_vline(x=50, line_color='black', line_width=1, line_dash='dash')

# DOTTED BLACK FRONTIER (original curve)
max_idx = df['return'].idxmax()
max_point = df.loc[max_idx]

x_max = max_point['win_rate']
y_max = max_point['return']

# Left part: constant max return
left_x = np.linspace(df['win_rate'].min(), x_max, 100)
left_y = np.full_like(left_x, y_max)

# Right part: sliding window
right_df = df[df['win_rate'] > x_max].sort_values(by='win_rate')
window_size = 40
step = 2

x_right = []
y_right = []

max_win_rate = right_df['win_rate'].max()

if not right_df.empty and not np.isnan(max_win_rate):

    for start in np.arange(x_max, max_win_rate, step):
        end = start + window_size
        segment = right_df[
            (right_df['win_rate'] >= start) &
            (right_df['win_rate'] < end)
        ]

        if not segment.empty:
            idx = segment['return'].idxmax()
            x_right.append(segment.loc[idx, 'win_rate'])
            y_right.append(segment.loc[idx, 'return'])

x_curve_black = np.concatenate([left_x, x_right])
y_curve_black = np.concatenate([left_y, y_right])

# Red shading
y_min_data = float(df['return'].min())
y_floor = y_min_data - 0.05 * (df['return'].max() - y_min_data)

fig.add_trace(
    go.Scatter(
        x=x_curve_black,
        y=[y_floor] * len(x_curve_black),
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    )
)

fig.add_trace(
    go.Scatter(
        x=x_curve_black,
        y=y_curve_black,
        mode="lines",
        line=dict(color="black", width=2, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(255,0,0,0.15)",
        name="Inefficient Region"
    )
)

# TRUE PARETO FRONTIER (BLUE LINE)
pareto_df = df[['win_rate','return','club_name']].rename(
    columns={"club_name_financial": "club_name"}
).dropna()

pareto_df = pareto_df.sort_values(by="win_rate", ascending=False)

pareto_points = []
max_ret_seen = -np.inf

for _, row in pareto_df.iterrows():
    if row['return'] > max_ret_seen:
        pareto_points.append(row)
        max_ret_seen = row['return']

pareto_df = pd.DataFrame(pareto_points).sort_values(by="win_rate")

# Global efficient teams
max_win = pareto_df.loc[pareto_df['win_rate'].idxmax()]
max_return = pareto_df.loc[pareto_df['return'].idxmax()]

# Blue Pareto line
fig.add_trace(go.Scatter(
    x=pareto_df["win_rate"],
    y=pareto_df["return"],
    mode="lines+markers",
    line=dict(color="blue", width=3),
    marker=dict(size=10, color="blue"),
    text=pareto_df["club_name"],
    hovertemplate="<b>%{text}</b><br>Pareto Efficient<br>Win Rate: %{x:.1f}%<br>Return: €%{y:,.0f}<extra></extra>",
    name="Pareto Frontier"
))

# Max win-rate (green)
fig.add_trace(go.Scatter(
    x=[max_win["win_rate"]],
    y=[max_win["return"]],
    mode="markers+text",
    marker=dict(size=14, color="green"),
    text=[max_win["club_name"]],
    textposition="top center",
    name="Max Win Rate"
))

# Max return (red)
fig.add_trace(go.Scatter(
    x=[max_return["win_rate"]],
    y=[max_return["return"]],
    mode="markers+text",
    marker=dict(size=14, color="red"),
    text=[max_return["club_name"]],
    textposition="top center",
    name="Max Return"
))
club_objetivo = df[df["club_name"] == target_club].iloc[0]

fig.add_trace(go.Scatter(
    x=[club_objetivo["win_rate"]],
    y=[club_objetivo["return"]],
    mode="markers+text",
    marker=dict(size=16, color="gold", symbol="star"),
    text=[club_objetivo["club_name"]],
    textposition="top center",
    name="Target Club"
))



# LAYOUT
fig.update_layout(
    template="plotly_white",
    title=f"Return vs Win Rate — Pareto Frontier)",
    xaxis_title="Win Rate (%)",
    yaxis_title="Return (€)",
    hovermode="closest",
    margin=dict(l=40, r=20, t=60, b=60),
)

fig.update_xaxes(tickformat=".0f")
fig.update_yaxes(tickformat=",.0f")
st.plotly_chart(fig, use_container_width=True)




# ==========================
# SLIDING FRONTIER
# ==========================
df = df_filtrado.copy()
df = df.dropna(subset=["win_rate", "return", "club_name"])

# Punto A: máxima rentabilidad
A_point = df.loc[df["return"].idxmax()]
W_A, R_A = float(A_point["win_rate"]), float(A_point["return"])

# Punto B: máximo win rate
B_point = df.loc[df["win_rate"].idxmax()]
W_B, R_B = float(B_point["win_rate"]), float(B_point["return"])


# Define R(W) = A - B * exp(γ W)
gamma = 0.05 

# Calculate B
den = np.exp(gamma * W_B) - np.exp(gamma * W_A)
B = (R_A - R_B) / den

# Calculate A
A_param = R_A + B * np.exp(gamma * W_A)


# Generate curve
x_curve = np.linspace(df["win_rate"].min(), df["win_rate"].max(), 400)
y_curve = A_param - B * np.exp(gamma * x_curve)

# Scatter base
fig = px.scatter(
    df,
    x="win_rate",
    y="return",
    hover_name="club_name",
    hover_data={"win_rate":":.1f", "return":":,.0f"},
    title="RoP Frontier Model — Exponential Efficient Frontier"
)

fig.update_traces(
    marker=dict(size=10, color="#777", opacity=0.8,
                line=dict(width=0.6, color="white")),
    selector=dict(mode="markers")
)

# Frontier Curve
fig.add_trace(go.Scatter(
    x=x_curve,
    y=y_curve,
    mode="lines",
    line=dict(color="blue", width=4),
    name="Efficient Frontier (Exponential, concave)"
))

# Point A: Max Return (red)
fig.add_trace(go.Scatter(
    x=[W_A], y=[R_A],
    mode="markers+text",
    text=[A_point["club_name"]],
    textposition="top center",
    marker=dict(size=14, color="red"),
    name="Max Return"
))

# Point B: Max Win Rate (green)
fig.add_trace(go.Scatter(
    x=[W_B], y=[R_B],
    mode="markers+text",
    text=[B_point["club_name"]],
    textposition="bottom center",
    marker=dict(size=14, color="green"),
    name="Max Win Rate"
))

# Reference liens (optional)
fig.add_hline(y=0, line_color='black', line_width=1, line_dash='dash')
fig.add_vline(x=50, line_color='black', line_width=1, line_dash='dash')

fig.update_layout(
    template="plotly_white",
    xaxis_title="Win Rate (%)",
    yaxis_title="Return (€)",
    hovermode="closest",
    showlegend=True
)

fig.update_xaxes(tickformat=".0f")
fig.update_yaxes(tickformat=",.0f")

# Red Area — INEFFICIENT REGION (Below frontier)
y_min_data = float(df["return"].min())
y_floor = y_min_data - 0.05 * (df["return"].max() - y_min_data)

# Lower line
fig.add_trace(go.Scatter(
    x=x_curve,
    y=[y_floor] * len(x_curve),
    mode="lines",
    line=dict(width=0),
    hoverinfo="skip",
    showlegend=False
))

# Red fill
fig.add_trace(go.Scatter(
    x=x_curve,
    y=y_curve,
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(255, 0, 0, 0.15)",
    hoverinfo="skip",
    showlegend=False
))

st.plotly_chart(fig, use_container_width=True)











# ==========================
# TABLA DE COMPARACIÓN DE KPIs
# ==========================
st.subheader("Comparación de Clubes — KPIs Relevantes")

# Selecciona las columnas más relevantes
kpi_columns = [
    "club_name",
    "league_name",
    "games_played",
    "wins",
    "win_rate",
    "total_income",
    "total_spent_transfer_fees",
    "total_spent_wages",
    "total_spent",
    "return",
    "is_profitable"
]

# Crear DataFrame filtrado solo con los KPIs seleccionados
df_kpis = df_filtrado[kpi_columns].sort_values("win_rate", ascending=False)
df_kpis["win_rate"] = df_kpis["win_rate"].map("{:.2f}%".format)
df_kpis["return"] = df_kpis["return"].map("{:,.0f}€".format)
df_kpis["total_income"] = df_kpis["total_income"].map("{:,.0f}€".format)
df_kpis["total_spent_transfer_fees"] = df_kpis["total_spent_transfer_fees"].map("{:,.0f}€".format)
df_kpis["total_spent_wages"] = df_kpis["total_spent_wages"].map("{:,.0f}€".format)
df_kpis["total_spent"] = df_kpis["total_spent"].map("{:,.0f}€".format)
df_kpis = df_kpis.reset_index(drop=True)
# Mostrar la tabla interactiva en Streamlit
st.dataframe(df_kpis, use_container_width=True)
