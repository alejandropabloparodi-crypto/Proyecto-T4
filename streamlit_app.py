import streamlit as st
import pandas as pd
import numpy as np
import time

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Tablero Integral T4 - Cloud",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ Tablero Control Integral Multiactivo (Proyecto T4)")
st.markdown("Matriz de diagnóstico en tiempo real con semáforos y validación de condiciones.")

# ==========================================
# 2. CONFIGURACIÓN DE ACTIVOS
# ==========================================
CONFIG_ACTIVOS = {
    "XAUUSD (Oro)": {
        "codigo": "XAUUSD",
        "atr_periodo": 14,
        "adx_minimo": 25.0,
        "spread_max_pips": 3.0,
        "factor_sl_atr": 1.5,
        "factor_tp_atr": 3.0,
        "lote": 0.01
    },
    "BTCUSD (Bitcoin)": {
        "codigo": "BTCUSD",
        "atr_periodo": 14,
        "adx_minimo": 28.0,
        "spread_max_pips": 15.0,
        "factor_sl_atr": 2.0,
        "factor_tp_atr": 4.0,
        "lote": 0.01
    },
    "USTEC (Nasdaq)": {
        "codigo": "USTEC",
        "atr_periodo": 14,
        "adx_minimo": 22.0,
        "spread_max_pips": 2.5,
        "factor_sl_atr": 1.2,
        "factor_tp_atr": 2.5,
        "lote": 0.1
    },
    "US30 (Dow Jones)": {
        "codigo": "US30",
        "atr_periodo": 14,
        "adx_minimo": 22.0,
        "spread_max_pips": 3.0,
        "factor_sl_atr": 1.2,
        "factor_tp_atr": 2.5,
        "lote": 0.1
    },
    "EURUSD (Euro/Dólar)": {
        "codigo": "EURUSD",
        "atr_periodo": 14,
        "adx_minimo": 20.0,
        "spread_max_pips": 1.2,
        "factor_sl_atr": 1.0,
        "factor_tp_atr": 2.0,
        "lote": 0.1
    }
}

# ==========================================
# 3. GENERADOR DE DATOS DE MERCADO
# ==========================================
def obtener_datos_simulados(simbolo, n=100):
    np.random.seed(int(time.time() * 10) % 1000 + len(simbolo))
    p_base = {"XAUUSD": 2650.0, "BTCUSD": 65000.0, "USTEC": 19800.0, "US30": 42000.0, "EURUSD": 1.0850}.get(simbolo, 100.0)
    retornos = np.random.normal(0.0001, 0.002, n)
    p = p_base * np.exp(np.cumsum(retornos))
    return pd.DataFrame({
        'close': p,
        'high': p * (1 + np.abs(np.random.normal(0, 0.001, n))),
        'low': p * (1 - np.abs(np.random.normal(0, 0.001, n)))
    })

def calcular_indicadores(df, periodo=14):
    df_c = df.copy()
    tr1 = df_c['high'] - df_c['low']
    tr2 = (df_c['high'] - df_c['close'].shift(1)).abs()
    tr3 = (df_c['low'] - df_c['close'].shift(1)).abs()
    df_c['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(periodo).mean()

    df_c['up_move'] = df_c['high'] - df_c['high'].shift(1)
    df_c['down_move'] = df_c['low'].shift(1) - df_c['low']
    df_c['plus_dm'] = np.where((df_c['up_move'] > df_c['down_move']) & (df_c['up_move'] > 0), df_c['up_move'], 0.0)
    df_c['minus_dm'] = np.where((df_c['down_move'] > df_c['up_move']) & (df_c['down_move'] > 0), df_c['down_move'], 0.0)
    
    plus_di = 100 * (df_c['plus_dm'].rolling(periodo).mean() / df_c['atr'])
    minus_di = 100 * (df_c['minus_dm'].rolling(periodo).mean() / df_c['atr'])
    sum_di = np.where((plus_di + minus_di) == 0, 1e-9, plus_di + minus_di)
    dx = (np.abs(plus_di - minus_di) / sum_di) * 100
    df_c['adx'] = pd.Series(dx).rolling(periodo).mean()

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

    return {
        "precio": u_m15['close'],
        "spread": spread_pips,
        "adx": adx_val,
        "dir_m15": dir_m15,
        "dir_h1": dir_h1,
        "cond_spread": cond_spread,
        "cond_adx": cond_adx,
        "cond_mtf": cond_mtf,
        "listo_compra": listo_compra,
        "listo_venta": listo_venta,
        "estado_general": estado_general
    }

# ==========================================
# 4. BARRA LATERAL
# ==========================================
st.sidebar.header("☁️ Servidor Web Active")
st.sidebar.success("Panel alojado en la Nube (Render)")

if st.sidebar.button("🔄 Actualizar Tablero"):
    st.rerun()

# ==========================================
# 5. TABLERO MATRIZ DE SEMÁFOROS
# ==========================================
st.subheader("🖥️ Matriz Integral de Semáforos y Lista de Condiciones")

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
                    st.markdown(f"### **{simbolo}** | `{diag['precio']:.2f}`")
                    st.markdown(f"**Estado General:** {diag['estado_general']}")
                    st.markdown("---")
                    
                    st.write("**Lista de Condiciones:**")
                    st.write(f"{'✅' if diag['cond_spread'] else '❌'} **Spread:** {diag['spread']:.1f}p (Máx: {cfg['spread_max_pips']}p)")
                    st.write(f"{'✅' if diag['cond_adx'] else '❌'} **Fuerza ADX:** {diag['adx']:.1f} (Mín: {cfg['adx_minimo']})")
                    st.write(f"{'✅' if diag['cond_mtf'] else '❌'} **Alineación MTF:** M15 ({diag['dir_m15']}) / H1 ({diag['dir_h1']})")
                    
                    st.markdown("---")

                    btn1, btn2 = st.columns(2)
                    with btn1:
                        st.button(f"🟢 COMPRAR", key=f"b_{simbolo}", disabled=not diag["listo_compra"])
                    with btn2:
                        st.button(f"🔴 VENTAR", key=f"s_{simbolo}", disabled=not diag["listo_venta"])

st.markdown("---")
