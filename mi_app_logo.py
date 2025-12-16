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
import mplcursors
from matplotlib.ticker import FuncFormatter


def load_logo(url, size=(55, 55)):
    try:
        r = requests.get(url, timeout=4)
        img = Image.open(BytesIO(r.content)).convert("RGBA")
        img = img.resize(size, Image.LANCZOS)
        return img
    except:
        return None


# FUNCIONES AUXILIARES
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

#df = pd.read_csv("club_data.csv")
df = pd.read_csv("all_leagues.csv")
print("Loaded data with shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df["domestic_competition_id"].unique())
print(df.head())
league_map = {
    "ES1": "LaLiga",
    "FR1": "Ligue 1",
    "GB1": "Premier League",
    "L1":  "Bundesliga",
    "IT1": "Serie A",
    "TR1": "Süper Lig",
    "SC1": "Scottish Premiership",
    "GR1": "Greek Super League",
    "BE1": "Belgian Pro League",
    "RU1": "Russian Premier League",
    "NL1": "Eredivisie",
    "DK1": "Danish Superliga",
    "PO1": "Primeira Liga",
    "UKR1": "Ukrainian Premier League"
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


# DATOS BASE
ligas = sorted(df["league_name"].dropna().unique().tolist())
todos_los_clubes = sorted(df["club_name"].dropna().unique().tolist())

st.title("Comparador de clubes")

col_izq, col_der = st.columns(2)

# COLUMNA IZQUIERDA
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

# COLUMNA DERECHA
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

    
    # FILTROS
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


# RESULTADOS
if df_filtrado.empty:
    st.warning("No se encontraron clubes que cumplan los criterios seleccionados.")

st.divider()
st.subheader("Resultados")

st.write(f"**Club a comparar:** {target_club}")
# hacer que saga la liga sin [' y ']
ligas_comparar_str = ", ".join(ligas_comparar) if ligas_comparar else "Todas"
st.write(f"**Ligas a comparar:** {ligas_comparar_str}")
st.write(f"**Número de clubes tras filtros:** {len(df_filtrado)}")

# PARETO FRONTIER PLOT

if not club_target_df.empty:
    # Comprobar si ya está en df_filtrado
    if target_club not in df_filtrado["club_name"].values:
        # Añadirlo al DataFrame filtrado
        df_filtrado = pd.concat([df_filtrado, club_target_df], ignore_index=True)
else:
    st.warning(f"No se encontró el club objetivo: {target_club}")


logo_df = df_filtrado.dropna(subset=["club_logo_url"])

####################################################
def get_logo_image(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        # *** REAL SIZE BOOST HERE ***
        img = img.resize((40, 40), Image.LANCZOS)

        return OffsetImage(img, zoom=1.0)  # zoom stays fixed
    except:
        return None
    
def thousands_formatter(x, pos):
    return f"{int(x):,}".replace(",", ".")

# BASE FIGURE (GRAY POINTS)
fig, ax = plt.subplots(figsize=(15, 10))

# REFERENCE LINES
ax.axhline(0, linestyle="--", color="black")
ax.axvline(50, linestyle="--", color="black")

# ORIGINAL BLACK FRONTIER
max_idx = df_filtrado['return'].idxmax()
max_point = df_filtrado.loc[max_idx]

x_max = max_point['win_rate']
y_max = max_point['return']

left_x = np.linspace(df_filtrado['win_rate'].min(), x_max, 100)
left_y = np.full_like(left_x, y_max)

right_df = df_filtrado[df_filtrado['win_rate'] > x_max].sort_values(by="win_rate")
window_size = 40
step = 2

x_right, y_right = [], []

for start in np.arange(x_max, right_df['win_rate'].max(), step):
    end = start + window_size
    seg = right_df[(right_df["win_rate"] >= start) &
                    (right_df["win_rate"] < end)]
    if not seg.empty:
        best = seg.loc[seg['return'].idxmax()]
        x_right.append(best["win_rate"])
        y_right.append(best["return"])

x_curve_black = np.concatenate([left_x, x_right])
y_curve_black = np.concatenate([left_y, y_right])

# RED INEFFICIENT REGION
y_min = float(df_filtrado['return'].min())
y_floor = y_min - 0.05 * (df_filtrado['return'].max() - y_min)

ax.fill_between(
    x_curve_black, y_curve_black, y_floor,
    color=(1, 0, 0, 0.18)
)

ax.plot(x_curve_black, y_curve_black, linestyle=":", color="black", linewidth=2)

# PARETO FRONTIER (BLUE)
pareto_df = df[['win_rate','return','club_name','club_logo_url']].dropna()
pareto_df = pareto_df.sort_values(by="win_rate", ascending=False)

pts = []
seen = -np.inf
for _, row in pareto_df.iterrows():
    if row["return"] > seen:
        pts.append(row)
        seen = row["return"]

pareto_df = pd.DataFrame(pts).sort_values(by="win_rate")

ax.plot(
    pareto_df["win_rate"],
    pareto_df["return"],
    "-o",
    linewidth=4,
    markersize=10,
    color="blue",
    label="Pareto Frontier"
)


# ===========================
# CLUB LOGOS — FIXED BIG SIZE
# ===========================
for _, row in df_filtrado.iterrows():
    img = get_logo_image(row['club_logo_url'])
    if img:
        ab = AnnotationBbox(
            img,
            (row["win_rate"], row["return"]),
            frameon=False,
            box_alignment=(0.5, 0.5)
        )
        ax.add_artist(ab)

# Y AXIS FULL FORMAT
ax.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

# LAYOUT
ax.set_title(f"Return vs Win Rate — Logos View)", fontsize=20)
ax.set_xlabel("Win Rate (%)", fontsize=16)
ax.set_ylabel("Return (€)", fontsize=16)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()
plt.tight_layout()
st.pyplot(fig)




####################################################
# RoP Frontier Model — Exponential Efficient Frontier
df_filtrado = df_filtrado.dropna(subset=["win_rate", "return", "club_name"])

# Identify point A (max return) & B (max win rate)
A_point = df_filtrado.loc[df_filtrado["return"].idxmax()]
B_point = df_filtrado.loc[df_filtrado["win_rate"].idxmax()]

W_A, R_A = float(A_point["win_rate"]), float(A_point["return"])
W_B, R_B = float(B_point["win_rate"]), float(B_point["return"])
gamma = 0.05
# R(W) = A - B * exp(gamma W)
den = np.exp(gamma * W_B) - np.exp(gamma * W_A)
B_param = (R_A - R_B) / den
A_param = R_A + B_param * np.exp(gamma * W_A)

x_curve = np.linspace(df_filtrado["win_rate"].min(), df_filtrado["win_rate"].max(), 300)
y_curve = A_param - B_param * np.exp(gamma * x_curve)

# Shift curve UP so it covers Pareto points
pareto_df = df_filtrado.sort_values("win_rate", ascending=False)
best_return = -np.inf
pareto_points = []

for _, r in pareto_df.iterrows():
    if r["return"] > best_return:
        pareto_points.append(r)
        best_return = r["return"]

pareto_df = pd.DataFrame(pareto_points)

pareto_df["curve_pred"] = A_param - B_param * np.exp(gamma * pareto_df["win_rate"])
upward_shift = max(0, max(pareto_df["return"] - pareto_df["curve_pred"]))
A_param_shifted = A_param + upward_shift

y_curve = A_param_shifted - B_param * np.exp(gamma * x_curve)
# Matplotlib figure
fig, ax = plt.subplots(figsize=(14, 10))

# Red inefficient region
y_min_league = float(df_filtrado["return"].min())
y_floor = y_min_league - 0.05 * (df_filtrado["return"].max() - y_min_league)

ax.fill_between(
    x_curve, y_curve, y_floor,
    color=(1, 0, 0, 0.18),
    label="Inefficient Region"
)

# Frontier curve (blue)
ax.plot(x_curve, y_curve, color="blue", linewidth=4, label="Efficient Frontier")

# Scatter using logos
for _, row in df_filtrado.iterrows():
    logo = load_logo(row["club_logo_url"], size=(55, 55))
    if logo is not None:
        oi = OffsetImage(logo, zoom=1)
        ab = AnnotationBbox(oi, (row["win_rate"], row["return"]), frameon=False)
        ax.add_artist(ab)

# Max return (red circle)
ax.scatter(W_A, R_A, s=300, color="red", edgecolors="black", zorder=5)
ax.text(W_A, R_A, A_point["club_name"], fontsize=11, ha="center", va="bottom")

# Max win rate (green circle)
ax.scatter(W_B, R_B, s=300, color="green", edgecolors="black", zorder=5)
ax.text(W_B, R_B, B_point["club_name"], fontsize=11, ha="center", va="top")

# Layout
ax.set_title(f"Efficient Frontier", fontsize=20)
ax.set_xlabel("Win Rate (%)", fontsize=15)
ax.set_ylabel("Return (€)", fontsize=15)
ax.grid(True, linestyle="--", alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
#lim










# TABLA DE COMPARACIÓN DE KPIs
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
