import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Simulador de Demanda y Recaudación – Ruta Viva",
    layout="wide"
)

# Paleta (la tuya)
COLOR_1 = "#002A5C"   # azul oscuro
COLOR_2 = "#017DC3"   # azul claro
COLOR_3 = "#005DAB"   # azul medio

st.title("Simulador de demanda y recaudación – Peaje Ruta Viva")
st.subheader("**Hecho por:** Oikonomics Consultora económica")
st.caption(
    "Aplicación basada en una tabla de probabilidades simuladas (modelo de elección discreta). "
    "Estima cantidades esperadas por ruta y tipo de vehículo, curvas de demanda (precio vs cantidad) "
    "y recaudación esperada a partir de una distancia promedio."
)

# =====================================================
# ESCENARIO BASE (para botón Reset)
# =====================================================
DEFAULTS = {
    "tpda_liv": 84461.6,
    "tpda_bus": 1648.8,
    "tpda_pes": 2654.4,
    "mult_cam": 6.0,
    "dist_km": 10.0,
    "pmin": 0.00,
    "pmax": 0.80,
    "npts": 17,
    "tipo_vehiculo": "Ambos",
    "ruta": "Ruta Viva"
}

# Inicializa session_state (solo si no existe)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =====================================================
# DATOS (EMBEBIDOS, SIN CARGA DE CSV)
# =====================================================
@st.cache_data
def tabla_probabilidades_base() -> pd.DataFrame:
    """
    Tabla base (tu simulación):
    pkm: tarifa base (USD/km) para livianos
    pr_rv: Prob(Ruta Viva)
    pr_mix: Prob(Mixto)
    pr_int: Prob(Interoceánica)
    """
    return pd.DataFrame({
        "pkm":   [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "pr_rv": [0.46, 0.39, 0.32, 0.25, 0.19, 0.14, 0.10, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00],
        "pr_mix":[0.25, 0.32, 0.39, 0.45, 0.51, 0.56, 0.60, 0.63, 0.65, 0.66, 0.67, 0.68, 0.68, 0.69, 0.69, 0.69, 0.69],
        "pr_int":[0.28, 0.29, 0.29, 0.30, 0.30, 0.30, 0.30, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31],
    })


def interp_clamp(x_grid: np.ndarray, y_grid: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Interpolación lineal con clamp en extremos."""
    return np.interp(xq, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])


def calcular_resultados(
    df_prob: pd.DataFrame,
    tarifa_grid_liv: np.ndarray,
    tpda_liv: float,
    tpda_cam: float,
    mult_cam: float,
    dist_km: float
) -> pd.DataFrame:
    """
    Calcula probabilidades interpoladas y cantidades esperadas (veh/día) por ruta y tipo de vehículo.
    Además calcula recaudación esperada (USD/día y USD/año) solo para Ruta Viva.
    """
    df_prob = df_prob.sort_values("pkm").drop_duplicates(subset=["pkm"]).reset_index(drop=True)

    x = df_prob["pkm"].to_numpy(dtype=float)
    pr_rv  = df_prob["pr_rv"].to_numpy(dtype=float)
    pr_mix = df_prob["pr_mix"].to_numpy(dtype=float)
    pr_int = df_prob["pr_int"].to_numpy(dtype=float)

    # Livianos: evalúan prob en tarifa base
    p_liv = tarifa_grid_liv
    liv_rv  = interp_clamp(x, pr_rv,  p_liv)
    liv_mix = interp_clamp(x, pr_mix, p_liv)
    liv_int = interp_clamp(x, pr_int, p_liv)

    # Camiones: evalúan prob en tarifa multiplicada
    p_cam = tarifa_grid_liv * mult_cam
    cam_rv  = interp_clamp(x, pr_rv,  p_cam)
    cam_mix = interp_clamp(x, pr_mix, p_cam)
    cam_int = interp_clamp(x, pr_int, p_cam)

    # Cantidades esperadas
    q_rv_liv  = tpda_liv * liv_rv
    q_mix_liv = tpda_liv * liv_mix
    q_int_liv = tpda_liv * liv_int

    q_rv_cam  = tpda_cam * cam_rv
    q_mix_cam = tpda_cam * cam_mix
    q_int_cam = tpda_cam * cam_int

    out = pd.DataFrame({
        "tarifa_usd_km_livianos": p_liv,
        "tarifa_usd_km_camiones": p_cam,
        "distancia_promedio_km": dist_km,

        "pr_rv_liv":  liv_rv,  "Q_rv_liv_veh_dia":  q_rv_liv,
        "pr_mix_liv": liv_mix, "Q_mix_liv_veh_dia": q_mix_liv,
        "pr_int_liv": liv_int, "Q_int_liv_veh_dia": q_int_liv,

        "pr_rv_cam":  cam_rv,  "Q_rv_cam_veh_dia":  q_rv_cam,
        "pr_mix_cam": cam_mix, "Q_mix_cam_veh_dia": q_mix_cam,
        "pr_int_cam": cam_int, "Q_int_cam_veh_dia": q_int_cam,
    })

    # Recaudación: solo Ruta Viva
    out["recaud_liv_usd_dia"] = out["Q_rv_liv_veh_dia"] * (out["tarifa_usd_km_livianos"] * dist_km)
    out["recaud_cam_usd_dia"] = out["Q_rv_cam_veh_dia"] * (out["tarifa_usd_km_camiones"] * dist_km)
    out["recaud_total_usd_dia"] = out["recaud_liv_usd_dia"] + out["recaud_cam_usd_dia"]

    # Anual (365)
    out["recaud_liv_usd_anio"] = out["recaud_liv_usd_dia"] * 365
    out["recaud_cam_usd_anio"] = out["recaud_cam_usd_dia"] * 365
    out["recaud_total_usd_anio"] = out["recaud_total_usd_dia"] * 365

    return out


def grafico_curva(
    x_q,
    y_p,
    titulo,
    xlab,
    ylab,
    nombre_serie,
    color,
    dash=None
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_q,
        y=y_p,
        mode="lines+markers",
        name=nombre_serie,
        line=dict(color=color, dash=dash) if dash else dict(color=color)
    ))

    fig.update_layout(
        title=titulo,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def grafico_recaudacion(
    x_tarifa_liv,
    y_val,
    titulo,
    ylab,
    color,
    dash=None
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_tarifa_liv,
        y=y_val,
        mode="lines+markers",
        name="Recaudación",
        line=dict(color=color, dash=dash) if dash else dict(color=color)
    ))
    fig.update_layout(
        title=titulo,
        xaxis_title="Tarifa livianos (USD por km)",
        yaxis_title=ylab,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


# =====================================================
# SIDEBAR (PARÁMETROS) + BOTÓN RESET
# =====================================================
st.sidebar.header("Parámetros")

st.sidebar.subheader("Escenario")
if st.sidebar.button("🔄 Resetear escenario base"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.experimental_rerun()

st.sidebar.caption("Restablece todos los parámetros al escenario base del estudio.")

st.sidebar.subheader("TPDA (vehículos/día)")
tpda_liv = st.sidebar.number_input("Livianos", min_value=0.0, step=100.0, format="%.1f", key="tpda_liv")
tpda_bus = st.sidebar.number_input("Buses", min_value=0.0, step=10.0, format="%.1f", key="tpda_bus")
tpda_pes = st.sidebar.number_input("Pesados", min_value=0.0, step=10.0, format="%.1f", key="tpda_pes")
tpda_cam = tpda_bus + tpda_pes
st.sidebar.caption(f"Camiones = Buses + Pesados = {tpda_cam:,.1f} veh/día")

st.sidebar.subheader("Regla tarifaria")
mult_cam = st.sidebar.number_input("Multiplicador camiones (vs livianos)", min_value=0.0, step=0.5, format="%.2f", key="mult_cam")

st.sidebar.subheader("Distancia (para recaudación)")
dist_km = st.sidebar.number_input("Distancia promedio (km)", min_value=0.1, step=0.5, format="%.2f", key="dist_km")
st.sidebar.caption("Ingreso por vehículo = tarifa (USD/km) × distancia (km)")

df_prob = tabla_probabilidades_base()

st.sidebar.subheader("Rango de tarifa (livianos)")
pmin = st.sidebar.number_input("Tarifa mínima (USD/km)", min_value=0.0, step=0.01, format="%.2f", key="pmin")
pmax = st.sidebar.number_input("Tarifa máxima (USD/km)", min_value=0.0, step=0.01, format="%.2f", key="pmax")
npts = st.sidebar.slider("Número de puntos (curva)", min_value=5, max_value=250, step=1, key="npts")

tarifa_grid = np.linspace(float(pmin), float(pmax), int(npts))

st.sidebar.subheader("Visualización")
tipo_vehiculo = st.sidebar.selectbox(
    "Tipo de vehículo",
    ["Livianos", "Camiones (buses + pesados)", "Ambos"],
    key="tipo_vehiculo"
)

ruta = st.sidebar.selectbox(
    "Ruta a graficar",
    ["Ruta Viva", "Mixto", "Interoceánica"],
    key="ruta"
)


# =====================================================
# CÁLCULO
# =====================================================
resultados = calcular_resultados(
    df_prob=df_prob,
    tarifa_grid_liv=tarifa_grid,
    tpda_liv=tpda_liv,
    tpda_cam=tpda_cam,
    mult_cam=mult_cam,
    dist_km=dist_km
)

# =====================================================
# DESCARGA (SIN TABLAS EN PANTALLA)
# =====================================================
st.subheader("Descarga de resultados")
st.caption("Descarga el CSV con las variables usadas en los gráficos (probabilidades, cantidades y recaudación).")

csv_bytes = resultados.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar resultados (CSV)",
    data=csv_bytes,
    file_name="resultados_ruta_viva.csv",
    mime="text/csv"
)


# =====================================================
# RESUMEN DE RECAUDACIÓN (KPIs)
# =====================================================
st.subheader("Recaudación esperada (Ruta Viva)")

idx_max = int(resultados["recaud_total_usd_dia"].idxmax())
tarifa_opt = float(resultados.loc[idx_max, "tarifa_usd_km_livianos"])
recaud_max_dia = float(resultados.loc[idx_max, "recaud_total_usd_dia"])
recaud_max_anio = float(resultados.loc[idx_max, "recaud_total_usd_anio"])

c1, c2, c3 = st.columns(3)
c1.metric("Tarifa livianos que maximiza recaudación", f"${tarifa_opt:,.2f} / km")
c2.metric("Recaudación máxima estimada", f"${recaud_max_dia:,.0f} / día")
c3.metric("Recaudación máxima estimada", f"${recaud_max_anio:,.0f} / año")

st.caption(
    "La recaudación se calcula como: Q_RV(tipo) × (tarifa_tipo USD/km × distancia_promedio km). "
    "Para camiones, la tarifa se multiplica automáticamente por el multiplicador definido."
)


# =====================================================
# GRÁFICOS: CURVAS DE DEMANDA (ESTILO PLOTLY)
# =====================================================
st.subheader("Curvas de demanda (precio vs cantidad)")

if ruta == "Ruta Viva":
    q_liv_col = "Q_rv_liv_veh_dia"
    q_cam_col = "Q_rv_cam_veh_dia"
    titulo_base = "Curva de demanda – Ruta Viva"
elif ruta == "Mixto":
    q_liv_col = "Q_mix_liv_veh_dia"
    q_cam_col = "Q_mix_cam_veh_dia"
    titulo_base = "Curva (implícita) – Mixto"
else:
    q_liv_col = "Q_int_liv_veh_dia"
    q_cam_col = "Q_int_cam_veh_dia"
    titulo_base = "Curva (implícita) – Interoceánica"

colA, colB = st.columns(2)

if tipo_vehiculo in ["Livianos", "Ambos"]:
    fig_liv = grafico_curva(
        x_q=resultados[q_liv_col],
        y_p=resultados["tarifa_usd_km_livianos"],
        titulo=f"{titulo_base} (Vehículos livianos)",
        xlab=f"Cantidad esperada ({ruta}) – livianos (veh/día)",
        ylab="Tarifa (USD por km)",
        nombre_serie="Livianos",
        color=COLOR_1,
        dash=None
    )
    colA.plotly_chart(fig_liv, use_container_width=True)

if tipo_vehiculo in ["Camiones (buses + pesados)", "Ambos"]:
    fig_cam = grafico_curva(
        x_q=resultados[q_cam_col],
        y_p=resultados["tarifa_usd_km_camiones"],
        titulo=f"{titulo_base} (Camiones: buses + pesados)",
        xlab=f"Cantidad esperada ({ruta}) – camiones (veh/día)",
        ylab="Tarifa (USD por km)",
        nombre_serie="Camiones",
        color=COLOR_3,
        dash="dash"
    )
    target_col = colB if tipo_vehiculo == "Ambos" else colA
    target_col.plotly_chart(fig_cam, use_container_width=True)

st.caption("Nota visual: si seleccionas 'Ambos', verás dos gráficos separados para evitar distorsión por escalas diferentes.")


# =====================================================
# GRÁFICOS: RECAUDACIÓN
# =====================================================
st.subheader("Curvas de recaudación (Ruta Viva)")

tab1, tab2 = st.tabs(["USD por día", "USD por año"])

with tab1:
    c1, c2, c3 = st.columns(3)

    if tipo_vehiculo in ["Livianos", "Ambos"]:
        fig = grafico_recaudacion(
            x_tarifa_liv=resultados["tarifa_usd_km_livianos"],
            y_val=resultados["recaud_liv_usd_dia"],
            titulo="Recaudación – Livianos (Ruta Viva)",
            ylab="Recaudación (USD por día)",
            color=COLOR_2,
            dash="dot"
        )
        c1.plotly_chart(fig, use_container_width=True)

    if tipo_vehiculo in ["Camiones (buses + pesados)", "Ambos"]:
        fig = grafico_recaudacion(
            x_tarifa_liv=resultados["tarifa_usd_km_livianos"],
            y_val=resultados["recaud_cam_usd_dia"],
            titulo="Recaudación – Camiones (Ruta Viva)",
            ylab="Recaudación (USD por día)",
            color=COLOR_3,
            dash="dash"
        )
        (c2 if tipo_vehiculo == "Ambos" else c1).plotly_chart(fig, use_container_width=True)

    fig = grafico_recaudacion(
        x_tarifa_liv=resultados["tarifa_usd_km_livianos"],
        y_val=resultados["recaud_total_usd_dia"],
        titulo="Recaudación – Total (Livianos + Camiones) – Ruta Viva",
        ylab="Recaudación (USD por día)",
        color=COLOR_1,
        dash=None
    )
    (c3 if tipo_vehiculo == "Ambos" else c2).plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2, c3 = st.columns(3)

    if tipo_vehiculo in ["Livianos", "Ambos"]:
        fig = grafico_recaudacion(
            x_tarifa_liv=resultados["tarifa_usd_km_livianos"],
            y_val=resultados["recaud_liv_usd_anio"],
            titulo="Recaudación anual – Livianos (Ruta Viva)",
            ylab="Recaudación (USD por año)",
            color=COLOR_2,
            dash="dot"
        )
        c1.plotly_chart(fig, use_container_width=True)

    if tipo_vehiculo in ["Camiones (buses + pesados)", "Ambos"]:
        fig = grafico_recaudacion(
            x_tarifa_liv=resultados["tarifa_usd_km_livianos"],
            y_val=resultados["recaud_cam_usd_anio"],
            titulo="Recaudación anual – Camiones (Ruta Viva)",
            ylab="Recaudación (USD por año)",
            color=COLOR_3,
            dash="dash"
        )
        (c2 if tipo_vehiculo == "Ambos" else c1).plotly_chart(fig, use_container_width=True)

    fig = grafico_recaudacion(
        x_tarifa_liv=resultados["tarifa_usd_km_livianos"],
        y_val=resultados["recaud_total_usd_anio"],
        titulo="Recaudación anual – Total (Livianos + Camiones) – Ruta Viva",
        ylab="Recaudación (USD por año)",
        color=COLOR_1,
        dash=None
    )
    (c3 if tipo_vehiculo == "Ambos" else c2).plotly_chart(fig, use_container_width=True)

st.caption(
    "Importante: la recaudación se calcula solo para Ruta Viva (porque es donde existe peaje). "
    "Las curvas por ruta alternativa se mantienen para demanda/asignación, no para recaudación."
)


