import streamlit as st
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Tablero Control T4 - Escalera & Multiactivo",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ Tablero Control Integral Multiactivo (Proyecto T4)")
st.markdown("Matriz de diagnóstico en tiempo real con semáforos, calculadora de riesgo y parámetros de **Escalera ATR**.")

# ==========================================
# 2. CONFIGURACIÓN DE ACTIVOS Y ESCALERA
# ==========================================
CONFIG_ACTIVOS = {
    "XAUUSD (Oro)": {
        "codigo": "XAUUSD",
        "atr_periodo": 14,
        "adx_minimo": 25.0,
        "spread_max_pips": 3.0,
        "factor_sl_inicial": 1.5,
        "factor_break_even": 1.0,
        "factor_paso_escalera": 1.0,
        "lote": 0.01
    },
    "BTCUSD (Bitcoin)": {
        "codigo": "BTCUSD",
        "atr_periodo": 14,
        "adx_minimo": 28.0,
        "spread_max_pips": 15.0,
        "factor_sl_inicial": 2.0,
        "factor_break_even": 1.2,
        "factor_paso_escalera": 1.2,
        "lote": 0.01
    },
    "USOIL (Petróleo WTI)": {
        "codigo": "USOIL",
        "atr_periodo": 14,
        "adx_minimo": 24.0,
        "spread_max_pips": 4.0,
        "factor_sl_inicial": 1.5,
        "factor_break_even": 1.0,
        "factor_paso_escalera": 1.0,
        "lote": 0.01
    }
}

# ==========================================
# 3. GENERADOR DE DATOS Y CÁLCULOS TÉCNICOS
# ==========================================
def obtener_datos_simulados(simbolo, n=100):
    np.random.seed(int(time.time() * 10) % 1000 + len(simbolo))
    
    # Precios base de referencia para el entorno visual web
    p_base = {
        "XAUUSD": 2680.0,   # Oro
        "BTCUSD": 65000.0,  # Bitcoin
        "USOIL": 72.50      # Petróleo WTI
    }.get(simbolo, 100.0)

    retornos = np.random.normal(0.0001, 0.002, n)
    p = p_base * np.exp(np.cumsum(retornos))
    return pd.DataFrame({
        'close': p,
        'high': p * (1 + np.abs(np.random.normal(0, 0.001, n))),
        'low': p * (1 - np.abs(np.random.normal(0, 0.001, n)))
    })

def calcular_indicadores(df, periodo=14):
    df_c = df.copy()
    # ATR
    tr1 = df_c['high'] - df_c['low']
    tr2 = (df_c['high'] - df_c['close'].shift(1)).abs()
    tr3 = (df_c['low'] - df_c['close'].shift(1)).abs()
    df_c['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(periodo).mean()

    # ADX
    df_c['up_move'] = df_c['high'] - df_c['high'].shift(1)
    df_c['down_move'] = df_c['low'].shift(1) - df_c['low']
    df_c['plus_dm'] = np.where((df_c['up_move'] > df_c['down_move']) & (df_c['up_move'] > 0), df_c['up_move'], 0.0)
    df_c['minus_dm'] = np.where((df_c['down_move'] > df_c['up_move']) & (df_c['down_move'] > 0), df_c['down_move'], 0.0)
    
    plus_di = 100 * (df_c['plus_dm'].rolling(periodo).mean() / df_c['atr'])
    minus_di = 100 * (df_c['minus_dm'].rolling(periodo).mean() / df_c['atr'])
    sum_di = np.where((plus_di + minus_di) == 0, 1e-9, plus_di + minus_di)
    dx = (np.abs(plus_di - minus_di) / sum_di) * 100
    df_c['adx'] = pd.Series(dx).rolling(periodo).mean()

    # EMAs
    df_c['ema_corta'] = df_c['close'].ewm(span=9, adjust=False).mean()
    df_c['ema_larga'] = df_c['close'].ewm(span=21, adjust=False).mean()
    return df_c

def diagnosticar_detallado(cfg):
    simbolo = cfg["codigo"]
    df_m15 = calcular_indicadores(obtener_datos_simulados(simbolo), cfg["atr_periodo"])
    df_h1 = calcular_indicadores(obtener_datos_simulados(simbolo), cfg["atr_periodo"])

    u_m15 = df_m15.iloc[-1]
    u_h1 = df_h1.iloc[-1]

    dir_m15 = "COMPRA" if u_m15['ema_corta'] > u_m15['ema_larga'] else "VENTA"
    dir_h1 = "COMPRA" if u_h1['ema_corta'] > u_h1['ema_larga'] else "VENTA"

    adx_val = u_m15['adx'] if not pd.isna(u_m15['adx']) else 0.0
    atr_val = u_m15['atr'] if not pd.isna(u_m15['atr']) else 1.0
    spread_pips = 1.2

    cond_spread = spread_pips <= cfg["spread_max_pips"]
    cond_adx = adx_val >= cfg["adx_minimo"]
    cond_mtf = (dir_m15 == dir_h1)

    listo_compra = cond_spread and cond_adx and cond_mtf and (dir_m15 == "COMPRA")
    listo_venta = cond_spread and cond_adx and cond_mtf and (dir_m15 == "VENTA")

    if listo_compra:
        estado_general = "🟢 LISTO PARA COMPRAR"
    elif listo_venta:
        estado_general = "🟢 LISTO PARA VENTAR"
    else:
        estado_general = "🔴 NO APTO / EN ESPERA"

    # Distancias en dólares calculadas con ATR
    sl_dist_USD = atr_val * cfg["factor_sl_inicial"]
    be_dist_USD = atr_val * cfg["factor_break_even"]
    paso_dist_USD = atr_val * cfg["factor_paso_escalera"]

    return {
        "precio": u_m15['close'],
        "spread": spread_pips,
        "adx": adx_val,
        "atr": atr_val,
        "dir_m15": dir_m15,
        "dir_h1": dir_h1,
        "cond_spread": cond_spread,
        "cond_adx": cond_adx,
        "cond_mtf": cond_mtf,
        "listo_compra": listo_compra,
        "listo_venta": listo_venta,
        "estado_general": estado_general,
        "sl_dist_USD": sl_dist_USD,
        "be_dist_USD": be_dist_USD,
        "paso_dist_USD": paso_dist_USD
    }

# ==========================================
# 4. BARRA LATERAL (CONTROLES)
# ==========================================
st.sidebar.header("☁️ Servidor Web Active")
st.sidebar.success("Panel alojado en la Nube (Render)")

if st.sidebar.button("🔄 Actualizar Tablero"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Calculadora de Gestión de Riesgo")
capital_cuenta = st.sidebar.number_input("Capital Cuenta ($USD):", value=1000.0, step=100.0)
riesgo_pct = st.sidebar.slider("Riesgo por operación (%):", 0.5, 3.0, 1.0)
riesgo_USD = capital_cuenta * (riesgo_pct / 100.0)
st.sidebar.info(f"Riesgo Máximo Permitido: **${riesgo_USD:.2f} USD**")

# ==========================================
# 5. TABLERO MATRIZ DE SEMÁFOROS Y ESCALERA
# ==========================================
st.subheader("🖥️ Matriz Integral de Semáforos & Parámetros de Escalera")

activos_lista = list(CONFIG_ACTIVOS.items())

for i in range(0, len(activos_lista), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(activos_lista):
            nombre, cfg = activos_lista[i + j]
            simbolo = cfg["codigo"]
            diag = diagnosticar_detallado(cfg)

            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"### **{simbolo}** | `${diag['precio']:.2f}`")
                    st.markdown(f"**Estado General:** {diag['estado_general']}")
                    st.markdown("---")
                    
                    st.write("**Filtros de Entrada:**")
                    st.write(f"{'✅' if diag['cond_spread'] else '❌'} **Spread:** {diag['spread']:.1f}p (Máx: {cfg['spread_max_pips']}p)")
                    st.write(f"{'✅' if diag['cond_adx'] else '❌'} **Fuerza ADX:** {diag['adx']:.1f} (Mín: {cfg['adx_minimo']})")
                    st.write(f"{'✅' if diag['cond_mtf'] else '❌'} **Alineación MTF:** M15 ({diag['dir_m15']}) / H1 ({diag['dir_h1']})")
                    
                    st.markdown("---")
                    st.write("**🧱 Configuración Escalera ATR:**")
                    st.write(f"• **Stop Loss Inicial:** -${diag['sl_dist_USD']:.2f} USD")
                    st.write(f"• **Activación Break-Even:** +${diag['be_dist_USD']:.2f} USD")
                    st.write(f"• **Ancho del Escalón:** +${diag['paso_dist_USD']:.2f} USD")

                    st.markdown("---")

                    btn1, btn2 = st.columns(2)
                    with btn1:
                        st.button(f"🟢 COMPRAR", key=f"b_{simbolo}", disabled=not diag["listo_compra"])
                    with btn2:
                        st.button(f"🔴 VENTAR", key=f"s_{simbolo}", disabled=not diag["listo_venta"])

st.markdown("---")
