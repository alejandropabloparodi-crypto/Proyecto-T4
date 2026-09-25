import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Tablero Integral de Semáforos - Proyecto T4",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ Tablero Control Integral Multiactivo (Proyecto T4)")
st.markdown("Matriz de diagnóstico en tiempo real con validación condición por condición y ejecución directa.")

# ==========================================
# 2. CONEXIÓN CON METATRADER 5
# ==========================================
def conectar_mt5():
    if not mt5.initialize():
        return False
    return True

mt5_conectado = conectar_mt5()

# ==========================================
# 3. CONFIGURACIÓN DE ACTIVOS
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
# 4. MOTOR DE LECTURA Y EVALUACIÓN DETALLADA
# ==========================================
def obtener_datos(simbolo, timeframe, n=100):
    tf_mt5 = mt5.TIMEFRAME_M15 if timeframe == "M15" else mt5.TIMEFRAME_H1
    if mt5_conectado and mt5.symbol_select(simbolo, True):
        rates = mt5.copy_rates_from_pos(simbolo, tf_mt5, 0, n)
        if rates is not None and len(rates) > 0:
            return pd.DataFrame(rates)
    
    np.random.seed(42)
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
    df_m15 = calcular_indicadores(obtener_datos(simbolo, "M15"), cfg["atr_periodo"])
    df_h1 = calcular_indicadores(obtener_datos(simbolo, "H1"), cfg["atr_periodo"])

    u_m15 = df_m15.iloc[-1]
    u_h1 = df_h1.iloc[-1]

    dir_m15 = "COMPRA" if u_m15['ema_corta'] > u_m15['ema_larga'] else "VENTA"
    dir_h1 = "COMPRA" if u_h1['ema_corta'] > u_h1['ema_larga'] else "VENTA"

    adx_val = u_m15['adx'] if not pd.isna(u_m15['adx']) else 0.0
    
    spread_pips = 1.0
    if mt5_conectado:
        info_tick = mt5.symbol_info_tick(simbolo)
        info_symbol = mt5.symbol_info(simbolo)
        if info_tick and info_symbol and info_symbol.point > 0:
            spread_pips = (info_tick.ask - info_tick.bid) / (info_symbol.point * 10)

    # Evaluación condición por condición
    cond_spread = spread_pips <= cfg["spread_max_pips"]
    cond_adx = adx_val >= cfg["adx_minimo"]
    cond_mtf = (dir_m15 == dir_h1)

    # Semáforo final
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
# 5. FUNCIÓN PARA ENVIAR ÓRDENES REALES
# ==========================================
def enviar_orden_mt5(simbolo, tipo_orden, lote, sl_dist_atr, tp_dist_atr):
    if not mt5_conectado:
        return False, "MT5 no está inicializado."

    mt5.symbol_select(simbolo, True)
    tick = mt5.symbol_info_tick(simbolo)
    if not tick:
        return False, f"Sin tick para {simbolo}"

    precio = tick.ask if tipo_orden == "COMPRA" else tick.bid
    tipo_mt5 = mt5.ORDER_TYPE_BUY if tipo_orden == "COMPRA" else mt5.ORDER_TYPE_SELL

    rates = mt5.copy_rates_from_pos(simbolo, mt5.TIMEFRAME_M15, 0, 20)
    df = pd.DataFrame(rates)
    tr = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    atr = tr.rolling(14).mean().iloc[-1]

    if tipo_orden == "COMPRA":
        sl = precio - (atr * sl_dist_atr)
        tp = precio + (atr * tp_dist_atr)
    else:
        sl = precio + (atr * sl_dist_atr)
        tp = precio - (atr * tp_dist_atr)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": simbolo,
        "volume": float(lote),
        "type": tipo_mt5,
        "price": float(precio),
        "sl": round(float(sl), 4 if "EUR" in simbolo else 2),
        "tp": round(float(tp), 4 if "EUR" in simbolo else 2),
        "deviation": 20,
        "magic": 10042026,
        "comment": "Proyecto T4 Tablero",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    resultado = mt5.order_send(request)
    if resultado.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Error MT5: {resultado.comment} ({resultado.retcode})"
    return True, f"¡Orden {tipo_orden} enviada! Ticket: #{resultado.order}"

# ==========================================
# 6. BARRA LATERAL (ESTADO CUENTA)
# ==========================================
st.sidebar.header("🔌 Estado MT5 / Exness")
if mt5_conectado:
    cuenta = mt5.account_info()
    if cuenta:
        st.sidebar.success(f"🟢 **CONECTADO**\n\nLogin: **{cuenta.login}**")
        st.sidebar.write(f"**Balance:** USD {cuenta.balance:.2f}")
        st.sidebar.write(f"**Equidad:** USD {cuenta.equity:.2f}")
else:
    st.sidebar.error("🔴 MT5 Desconectado.")

if st.sidebar.button("🔄 Actualizar Tablero"):
    st.rerun()

# ==========================================
# 7. TABLERO MATRIZ DE SEMÁFOROS (GRILLA)
# ==========================================
st.subheader("🖥️ Matriz Integral de Semáforos y Lista de Condiciones")

activos_lista = list(CONFIG_ACTIVOS.items())

# Distribución en Filas de 3 Activos
for i in range(0, len(activos_lista), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(activos_lista):
            nombre, cfg = activos_lista[i + j]
            simbolo = cfg["codigo"]
            diag = diagnosticar_detallado(cfg)

            with cols[j]:
                # Contenedor visual de la tarjeta
                with st.container(border=True):
                    st.markdown(f"### **{simbolo}** | `{diag['precio']:.2f}`")
                    st.markdown(f"**Estado General:** {diag['estado_general']}")
                    st.markdown("---")
                    
                    # Lista de Chequeo de Condiciones
                    st.write("**Lista de Condiciones:**")
                    st.write(f"{'✅' if diag['cond_spread'] else '❌'} **Spread:** {diag['spread']:.1f}p (Máx: {cfg['spread_max_pips']}p)")
                    st.write(f"{'✅' if diag['cond_adx'] else '❌'} **Fuerza ADX:** {diag['adx']:.1f} (Mín: {cfg['adx_minimo']})")
                    st.write(f"{'✅' if diag['cond_mtf'] else '❌'} **Alineación MTF:** M15 ({diag['dir_m15']}) / H1 ({diag['dir_h1']})")
                    
                    st.markdown("---")

                    # Botones inteligentes de Compra / Venta
                    btn1, btn2 = st.columns(2)
                    
                    with btn1:
                        # Se habilita si el semáforo aprueba COMPRA
                        if st.button(f"🟢 COMPRAR", key=f"b_{simbolo}", disabled=not diag["listo_compra"]):
                            exito, msg = enviar_orden_mt5(simbolo, "COMPRA", cfg["lote"], cfg["factor_sl_atr"], cfg["factor_tp_atr"])
                            if exito:
                                st.success(msg)
                            else:
                                st.error(msg)
                            st.rerun()

                    with btn2:
                        # Se habilita si el semáforo aprueba VENTA
                        if st.button(f"🔴 VENTAR", key=f"s_{simbolo}", disabled=not diag["listo_venta"]):
                            exito, msg = enviar_orden_mt5(simbolo, "VENTA", cfg["lote"], cfg["factor_sl_atr"], cfg["factor_tp_atr"])
                            if exito:
                                st.success(msg)
                            else:
                                st.error(msg)
                            st.rerun()

st.markdown("---")

# ==========================================
# 8. RESUMEN DE POSICIONES ABIERTAS EN MT5
# ==========================================
if mt5_conectado:
    pos_reales = mt5.positions_get()
    if pos_reales:
        st.subheader("📍 Posiciones Abiertas Reales en Exness")
        datos_p = []
        for p in pos_reales:
            t_str = "COMPRA" if p.type == mt5.ORDER_TYPE_BUY else "VENTA"
            datos_p.append({
                "Ticket": p.ticket,
                "Símbolo": p.symbol,
                "Tipo": t_str,
                "Lote": p.volume,
                "Precio Entrada": p.price_open,
                "Stop Loss": p.sl,
                "Take Profit": p.tp,
                "Ganancia USD": p.profit
            })
        st.dataframe(pd.DataFrame(datos_p), use_container_width=True)
