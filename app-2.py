
# ============================================================
#  ML TRADING DASHBOARD — Z-Score + Bollinger + MTF + ML
#  Actifs : XAUUSD | BTC/USD | CL WTI | 6E Euro Futures
#  Auteur  : Claude / Anthropic
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="ML Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS GLOBAL ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #e0e6f0;
}
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%); }

/* HEADER */
.dashboard-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.1rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 10px 0;
    letter-spacing: 3px;
    text-shadow: none;
}
.subtitle {
    text-align: center;
    color: #667eea;
    font-size: 0.95rem;
    letter-spacing: 2px;
    margin-bottom: 20px;
}

/* CARDS */
.price-card {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
    box-shadow: 0 4px 20px rgba(0,212,255,0.08);
    transition: all 0.3s ease;
}
.price-card:hover { border-color: #00d4ff; box-shadow: 0 4px 25px rgba(0,212,255,0.18); }
.price-main {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4ff;
}
.price-label { font-size: 0.8rem; color: #667eea; letter-spacing: 2px; text-transform: uppercase; }
.price-change-pos { color: #00ff88; font-size: 1rem; font-weight: 600; }
.price-change-neg { color: #ff4d6d; font-size: 1rem; font-weight: 600; }

/* SIGNAL BADGE */
.signal-buy  { background: linear-gradient(90deg,#00ff88,#00cc6a); color:#000; padding:6px 18px; border-radius:20px; font-weight:700; font-size:0.9rem; }
.signal-sell { background: linear-gradient(90deg,#ff4d6d,#cc0033); color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:0.9rem; }
.signal-neutral { background: linear-gradient(90deg,#667eea,#764ba2); color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:0.9rem; }

/* KILL ZONE */
.killzone-active { background:linear-gradient(90deg,rgba(255,107,53,0.2),rgba(255,107,53,0.05)); border-left:4px solid #ff6b35; border-radius:8px; padding:12px 18px; margin:4px 0; }
.killzone-next   { background:linear-gradient(90deg,rgba(123,47,247,0.15),rgba(123,47,247,0.03)); border-left:4px solid #7b2ff7; border-radius:8px; padding:12px 18px; margin:4px 0; }
.killzone-label  { font-family:'Orbitron',sans-serif; font-size:0.8rem; color:#ff6b35; letter-spacing:2px; }
.killzone-label-next { font-family:'Orbitron',sans-serif; font-size:0.8rem; color:#7b2ff7; letter-spacing:2px; }

/* PREDICTION MODAL */
.pred-window {
    background: linear-gradient(135deg, #0d1526 0%, #1a0a2e 100%);
    border: 2px solid #7b2ff7;
    border-radius: 16px;
    padding: 24px;
    margin: 10px 0;
    animation: pulse-border 2s infinite;
    box-shadow: 0 0 30px rgba(123,47,247,0.3);
}
@keyframes pulse-border {
    0%   { box-shadow: 0 0 20px rgba(123,47,247,0.3); }
    50%  { box-shadow: 0 0 40px rgba(123,47,247,0.6); }
    100% { box-shadow: 0 0 20px rgba(123,47,247,0.3); }
}
.pred-title { font-family:'Orbitron',sans-serif; color:#7b2ff7; font-size:1rem; letter-spacing:2px; }
.metric-row { display:flex; justify-content:space-between; margin:8px 0; border-bottom:1px solid #1e3a5f; padding-bottom:6px; }
.metric-key   { color:#667eea; font-size:0.85rem; }
.metric-val   { color:#00d4ff; font-weight:600; font-size:0.85rem; }

/* MTF TABLE */
.mtf-table { border-radius:10px; overflow:hidden; }
.stDataFrame { background:#111827 !important; }

/* SECTION TITLES */
.section-title {
    font-family:'Orbitron',sans-serif;
    font-size:0.9rem;
    color:#00d4ff;
    letter-spacing:3px;
    text-transform:uppercase;
    border-bottom:1px solid #1e3a5f;
    padding-bottom:8px;
    margin:16px 0 12px 0;
}

/* SIDEBAR */
.css-1d391kg { background:#0d1526 !important; }
section[data-testid="stSidebar"] { background:#0d1526; border-right:1px solid #1e3a5f; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#0a0e1a; }
::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTES ─────────────────────────────────────────────
ASSETS = {
    "🥇 XAUUSD (Or)"       : "GC=F",
    "₿ BTC/USD"            : "BTC-USD",
    "🛢️ CL WTI (Pétrole)"  : "CL=F",
    "💶 6E (Euro Futures)"  : "EURUSD=X",
}

TIMEFRAMES = {
    "15 min": ("15m",  "5d"),
    "30 min": ("30m",  "5d"),
    "1 Heure": ("1h",  "30d"),
}

# Kill Zones (UTC)
KILL_ZONES = {
    "🌏 Tokyo KZ"        : (0,  3),
    "🏦 London Open KZ"  : (7, 10),
    "🗽 New York KZ"     : (13, 16),
    "🔒 London Close KZ" : (15, 17),
    "🌙 Sydney KZ"       : (22, 24),
}

UTC = pytz.utc


# ════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
        return pd.DataFrame()


def calc_indicators(df: pd.DataFrame, bb_window=20, bb_std=2.0, z_window=20) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]

    # ── Bollinger Bands
    d["BB_mid"]   = c.rolling(bb_window).mean()
    d["BB_std"]   = c.rolling(bb_window).std()
    d["BB_upper"] = d["BB_mid"] + bb_std * d["BB_std"]
    d["BB_lower"] = d["BB_mid"] - bb_std * d["BB_std"]
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / d["BB_mid"]
    d["BB_pos"]   = (c - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"])

    # ── Z-Score glissant
    d["Z_score"] = (c - d["BB_mid"]) / d["BB_std"]

    # ── RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    d["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD
    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    d["MACD"]  = ema12 - ema26
    d["Signal"]= d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["Signal"]

    # ── ATR
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - c.shift()).abs()
    lc  = (d["Low"]  - c.shift()).abs()
    d["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # ── Stochastique
    low14  = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["Stoch_K"] = 100 * (c - low14) / (high14 - low14)
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    # ── EMA 50 / 200
    d["EMA50"]  = c.ewm(span=50,  adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()

    # ── Momentum
    d["Momentum"] = c.pct_change(5) * 100

    return d.dropna()


def calc_pivots(df: pd.DataFrame) -> dict:
    """Pivot Points Classiques basés sur la bougie précédente"""
    if len(df) < 2:
        return {}
    prev = df.iloc[-2]
    H, L, C = prev["High"], prev["Low"], prev["Close"]
    P  = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    R3 = H + 2 * (P - L)
    S3 = L - 2 * (H - P)
    return {"PP": P, "R1": R1, "R2": R2, "R3": R3,
            "S1": S1, "S2": S2, "S3": S3}


def get_killzone_status() -> dict:
    """Retourne KZ active + prochaine KZ"""
    now_utc = datetime.now(UTC)
    h = now_utc.hour
    active, next_kz, next_start = None, None, None

    for name, (start, end) in KILL_ZONES.items():
        if start <= h < end:
            active = name
            break

    # Prochaine KZ
    min_diff = 9999
    for name, (start, end) in KILL_ZONES.items():
        diff = (start - h) % 24
        if diff == 0:
            continue
        if diff < min_diff:
            min_diff = diff
            next_kz   = name
            next_start = (now_utc + timedelta(hours=diff)).strftime("%H:%M UTC")

    return {"active": active, "next": next_kz, "next_time": next_start,
            "current_time": now_utc.strftime("%H:%M UTC")}


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features ML et le label binaire"""
    d = df.copy()
    d["Target"] = (d["Close"].shift(-3) > d["Close"]).astype(int)
    features = ["Z_score", "RSI", "BB_pos", "BB_width",
                "MACD_hist", "Stoch_K", "Stoch_D",
                "ATR", "Momentum"]
    return d[features + ["Target"]].dropna()


def train_ml(df: pd.DataFrame):
    """Entraîne un GradientBoosting sur les features"""
    data = build_ml_features(df)
    if len(data) < 60:
        return None, None, None

    X = data.drop("Target", axis=1)
    y = data["Target"]
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(n_estimators=80,
                                              max_depth=3,
                                              learning_rate=0.1,
                                              random_state=42))
    ])
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    return model, X.columns.tolist(), round(acc * 100, 1)


def predict_signal(model, df: pd.DataFrame, feature_cols):
    """Retourne la prédiction + probabilité pour la dernière bougie"""
    if model is None:
        return None, None
    features = ["Z_score", "RSI", "BB_pos", "BB_width",
                "MACD_hist", "Stoch_K", "Stoch_D", "ATR", "Momentum"]
    last = df[features].dropna().iloc[-1:]
    if last.empty:
        return None, None
    pred  = model.predict(last)[0]
    proba = model.predict_proba(last)[0]
    return pred, proba


def mtf_analysis(ticker: str) -> pd.DataFrame:
    """Analyse Multi-TimeFrame"""
    rows = []
    for tf_name, (interval, period) in TIMEFRAMES.items():
        df = fetch_data(ticker, interval, period)
        if df.empty or len(df) < 30:
            rows.append({
                "TF": tf_name, "Prix": "N/A", "Z-Score": "N/A",
                "BB": "N/A", "RSI": "N/A", "MACD": "N/A",
                "Tendance": "N/A", "Signal": "⚪ N/A"
            })
            continue
        d = calc_indicators(df)
        last = d.iloc[-1]
        z   = round(last["Z_score"], 2)
        rsi = round(last["RSI"], 1)
        bb  = round(last["BB_pos"], 2)
        macd_h = last["MACD_hist"]
        close  = round(last["Close"], 4)

        # Tendance EMA
        trend = "↗ Haussier" if last["EMA50"] > last["EMA200"] else "↘ Baissier"

        # Signal composite
        score = 0
        if z < -1.5:  score += 2
        if z >  1.5:  score -= 2
        if rsi < 40:  score += 1
        if rsi > 60:  score -= 1
        if bb < 0.25: score += 1
        if bb > 0.75: score -= 1
        if macd_h > 0: score += 1
        else:          score -= 1

        if score >= 3:   sig = "🟢 ACHAT"
        elif score <= -3: sig = "🔴 VENTE"
        else:             sig = "⚪ NEUTRE"

        bb_label = "Bas" if bb < 0.33 else ("Haut" if bb > 0.66 else "Mid")
        rows.append({
            "TF": tf_name,
            "Prix": close,
            "Z-Score": z,
            "BB Position": bb_label,
            "RSI": rsi,
            "MACD": "↑" if macd_h > 0 else "↓",
            "Tendance": trend,
            "Signal ML": sig,
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
# GRAPHIQUE PRINCIPAL
# ════════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, pivots: dict, asset_name: str) -> go.Figure:
    last_n = min(200, len(df))
    d = df.tail(last_n)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.18, 0.16, 0.16],
        subplot_titles=(
            f"{asset_name} — Chandeliers + Bollinger + Pivots",
            "Z-Score",
            "RSI + Stochastique",
            "MACD"
        )
    )

    # ── Chandeliers
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"],
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4d6d",
        name="Prix", showlegend=False
    ), row=1, col=1)

    # ── Bollinger Bands
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_upper"], name="BB Sup",
        line=dict(color="rgba(0,212,255,0.5)", dash="dash", width=1),
        showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_mid"], name="BB Mid",
        line=dict(color="rgba(0,212,255,0.8)", width=1),
        showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_lower"], name="BB Inf",
        line=dict(color="rgba(0,212,255,0.5)", dash="dash", width=1),
        fill="tonexty", fillcolor="rgba(0,212,255,0.04)",
        showlegend=True), row=1, col=1)

    # ── EMA 50 / 200
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA50"], name="EMA 50",
        line=dict(color="#ff9500", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA200"], name="EMA 200",
        line=dict(color="#ff3b7a", width=1.2)), row=1, col=1)

    # ── Pivot Points (lignes horizontales)
    colors_piv = {"PP":"#ffffff","R1":"#ff6b6b","R2":"#ff3b3b","R3":"#cc0000",
                  "S1":"#6bff6b","S2":"#3bff3b","S3":"#00cc00"}
    for lev, val in pivots.items():
        fig.add_hline(y=val, line=dict(color=colors_piv.get(lev,"#888"),
                                        dash="dot", width=1),
                      annotation_text=f"  {lev}: {val:.4f}",
                      annotation_font_color=colors_piv.get(lev,"#888"),
                      annotation_font_size=9, row=1, col=1)

    # ── Z-Score
    z = d["Z_score"]
    fig.add_trace(go.Scatter(x=d.index, y=z, name="Z-Score",
        line=dict(color="#7b2ff7", width=1.5)), row=2, col=1)
    for val, col in [(2,"rgba(255,77,109,0.3)"),(-2,"rgba(0,255,136,0.3)")]:
        fig.add_hline(y=val, line=dict(color=col.replace("0.3","1"), dash="dash", width=1),
                      row=2, col=1)
    fig.add_hrect(y0=2, y1=4, fillcolor="rgba(255,77,109,0.07)", row=2, col=1)
    fig.add_hrect(y0=-4, y1=-2, fillcolor="rgba(0,255,136,0.07)", row=2, col=1)
    # Z-Score coloré
    colors_z = ["#ff4d6d" if v > 0 else "#00ff88" for v in z]
    fig.add_trace(go.Bar(x=d.index, y=z, name="Z Hist",
        marker_color=colors_z, opacity=0.4, showlegend=False), row=2, col=1)

    # ── RSI
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
        line=dict(color="#00d4ff", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Stoch_K"], name="Stoch K",
        line=dict(color="#ffd700", width=1, dash="dot")), row=3, col=1)
    for lv, cl in [(70,"rgba(255,77,109,0.3)"),(30,"rgba(0,255,136,0.3)")]:
        fig.add_hline(y=lv, line=dict(color=cl.replace("0.3","0.7"), dash="dash", width=1),
                      row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,77,109,0.05)", row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,136,0.05)",  row=3, col=1)

    # ── MACD
    macd_colors = ["#00ff88" if v >= 0 else "#ff4d6d" for v in d["MACD_hist"]]
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_hist"], name="MACD Hist",
        marker_color=macd_colors, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
        line=dict(color="#7b2ff7", width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["Signal"], name="Signal",
        line=dict(color="#ff9500", width=1.2)), row=4, col=1)

    # ── Layout
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1526",
        font=dict(family="Rajdhani", color="#e0e6f0", size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="rgba(10,14,26,0.8)", bordercolor="#1e3a5f",
            borderwidth=1, font=dict(size=10)
        ),
        height=820,
        margin=dict(l=50, r=30, t=40, b=20),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#1a2540", zerolinecolor="#1e3a5f", row=i, col=1)
        fig.update_yaxes(gridcolor="#1a2540", zerolinecolor="#1e3a5f", row=i, col=1)

    return fig


# ════════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ════════════════════════════════════════════════════════════

def main():
    # ── HEADER
    st.markdown('<div class="dashboard-title">⚡ ML TRADING DASHBOARD ⚡</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Z-Score · Bollinger · MTF · Pivots · Kill Zones · ML Prediction</div>', unsafe_allow_html=True)

    # ── SIDEBAR
    with st.sidebar:
        st.markdown('<div class="section-title">⚙ CONFIGURATION</div>', unsafe_allow_html=True)

        asset_label = st.selectbox("Actif principal", list(ASSETS.keys()))
        ticker = ASSETS[asset_label]

        tf_label = st.selectbox("Timeframe principal", list(TIMEFRAMES.keys()))
        interval, period = TIMEFRAMES[tf_label]

        bb_window = st.slider("Fenêtre Bollinger", 10, 50, 20)
        bb_std    = st.slider("Std Bollinger", 1.0, 3.0, 2.0, step=0.1)
        z_window  = st.slider("Fenêtre Z-Score", 10, 50, 20)
        z_seuil   = st.slider("Seuil Z-Score signal", 1.0, 3.0, 2.0, step=0.25)

        auto_refresh = st.checkbox("🔄 Auto-actualisation (60s)", value=True)
        st.markdown("---")
        st.markdown('<div class="section-title">📡 STATUT</div>', unsafe_allow_html=True)
        now_str = datetime.now(UTC).strftime("%H:%M:%S UTC")
        st.markdown(f"🕐 **{now_str}**")

        if auto_refresh:
            st.markdown("🟢 Live actif")
        else:
            st.markdown("🔴 Live inactif")

        if st.button("🔄 Actualiser maintenant"):
            st.cache_data.clear()

    # ── FETCH DATA
    with st.spinner("Chargement des données..."):
        df_raw = fetch_data(ticker, interval, period)

    if df_raw.empty:
        st.error("❌ Impossible de charger les données. Vérifiez votre connexion.")
        return

    df = calc_indicators(df_raw, bb_window, bb_std, z_window)
    if df.empty:
        st.error("❌ Données insuffisantes pour calculer les indicateurs.")
        return

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    pivots = calc_pivots(df)

    # Prix & variation
    price_now   = last["Close"]
    price_prev  = prev["Close"]
    price_open  = df_raw.iloc[-1]["Open"]
    variation   = ((price_now - price_prev) / price_prev) * 100
    var_day     = ((price_now - price_open) / price_open) * 100

    # ── LIGNE PRIX EN TEMPS RÉEL
    st.markdown('<div class="section-title">💹 PRIX EN TEMPS RÉEL</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    var_cls_1 = "price-change-pos" if variation >= 0 else "price-change-neg"
    var_cls_2 = "price-change-pos" if var_day >= 0   else "price-change-neg"
    arrow1 = "▲" if variation >= 0 else "▼"
    arrow2 = "▲" if var_day >= 0   else "▼"

    # Format prix
    def fmt(v):
        return f"{v:,.4f}" if v < 100 else f"{v:,.2f}"

    with c1:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">ACTIF</div>
            <div class="price-main">{fmt(price_now)}</div>
            <div class="{var_cls_1}">{arrow1} {abs(variation):.3f}% (bougie)</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">VARIATION JOURNÉE</div>
            <div class="price-main" style="font-size:1.4rem">{arrow2} {abs(var_day):.3f}%</div>
            <div style="color:#667eea;font-size:0.85rem">Open: {fmt(price_open)}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        z_val = last["Z_score"]
        z_col = "#ff4d6d" if z_val > z_seuil else ("#00ff88" if z_val < -z_seuil else "#00d4ff")
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Z-SCORE ACTUEL</div>
            <div class="price-main" style="color:{z_col}">{z_val:.3f}</div>
            <div style="color:#667eea;font-size:0.85rem">RSI: {last['RSI']:.1f} | ATR: {last['ATR']:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        bb_pct = last["BB_pos"] * 100
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">BOLLINGER POSITION</div>
            <div class="price-main" style="font-size:1.4rem">{bb_pct:.1f}%</div>
            <div style="color:#667eea;font-size:0.85rem">Width: {last['BB_width']*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── BARRE PROGRESSION BB
    bb_pos_pct = int(np.clip(last["BB_pos"] * 100, 0, 100))
    col_bb = "#ff4d6d" if bb_pos_pct > 75 else ("#00ff88" if bb_pos_pct < 25 else "#00d4ff")
    st.markdown(f"""
    <div style="margin:8px 0; background:#1a2540; border-radius:8px; padding:6px 12px;">
        <div style="font-size:0.75rem;color:#667eea;margin-bottom:4px">📊 Position dans Bollinger Bands</div>
        <div style="background:#0a0e1a;border-radius:6px;height:14px;overflow:hidden">
            <div style="width:{bb_pos_pct}%;background:linear-gradient(90deg,#00ff88,{col_bb});
                        height:100%;border-radius:6px;transition:width 0.5s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#445;">
            <span>S3</span><span>S2</span><span>S1</span><span>PP</span><span>R1</span><span>R2</span><span>R3</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── GRAPHIQUE PRINCIPAL
    st.markdown('<div class="section-title">📈 GRAPHIQUE ANALYSE TECHNIQUE</div>', unsafe_allow_html=True)
    chart = build_chart(df, pivots, f"{asset_label} [{tf_label}]")
    chart_placeholder = st.plotly_chart(chart, use_container_width=True, key="main_chart")

    # ─── COLONNES PRINCIPALES ────────────────────────────────
    left, right = st.columns([1.4, 1])

    # ── COLONNE GAUCHE : MTF + PIVOTS
    with left:
        # ── MTF ANALYSIS TABLE
        st.markdown('<div class="section-title">🔭 ANALYSE MULTI-TIMEFRAME</div>', unsafe_allow_html=True)
        with st.spinner("Analyse MTF en cours..."):
            mtf_df = mtf_analysis(ticker)

        def color_signal(val):
            if "ACHAT" in str(val):  return "background-color:#002200;color:#00ff88;font-weight:bold"
            if "VENTE" in str(val):  return "background-color:#220000;color:#ff4d6d;font-weight:bold"
            return "background-color:#111827;color:#667eea"

        def color_trend(val):
            if "Haussier" in str(val): return "color:#00ff88"
            if "Baissier" in str(val): return "color:#ff4d6d"
            return ""

        styled = (mtf_df.style
                  .applymap(color_signal, subset=["Signal ML"])
                  .applymap(color_trend,  subset=["Tendance"])
                  .set_properties(**{"background-color":"#111827","color":"#e0e6f0","border":"1px solid #1e3a5f"})
                  .set_table_styles([{"selector":"th","props":[("background-color","#0d1526"),
                                                                ("color","#00d4ff"),("font-family","Orbitron,sans-serif"),
                                                                ("font-size","0.75rem"),("letter-spacing","1px"),
                                                                ("border","1px solid #1e3a5f")]}]))
        st.dataframe(styled, use_container_width=True, height=180)

        # ── PIVOT POINTS
        st.markdown('<div class="section-title">📌 POINTS PIVOTS CLASSIQUES</div>', unsafe_allow_html=True)
        if pivots:
            pivot_cols = st.columns(7)
            labels = ["S3","S2","S1","PP","R1","R2","R3"]
            p_vals  = [pivots.get(k, 0) for k in labels]
            colors_p = ["#00cc00","#3bff3b","#6bff6b","#ffffff","#ff6b6b","#ff3b3b","#cc0000"]
            for i, (lbl, val, col) in enumerate(zip(labels, p_vals, colors_p)):
                with pivot_cols[i]:
                    diff = ((price_now - val) / val) * 100
                    d_str = f"{diff:+.2f}%"
                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid {col}33;border-radius:8px;
                                padding:8px;text-align:center;margin:2px;">
                        <div style="color:{col};font-family:Orbitron;font-size:0.7rem;font-weight:700">{lbl}</div>
                        <div style="color:#e0e6f0;font-size:0.85rem;font-weight:600">{fmt(val)}</div>
                        <div style="color:#667eea;font-size:0.7rem">{d_str}</div>
                    </div>""", unsafe_allow_html=True)

    # ── COLONNE DROITE : Kill Zones + Prédiction ML
    with right:
        # ── KILL ZONES
        st.markdown('<div class="section-title">🎯 KILL ZONES</div>', unsafe_allow_html=True)
        kz = get_killzone_status()

        if kz["active"]:
            st.markdown(f"""
            <div class="killzone-active">
                <div class="killzone-label">🔥 KILL ZONE ACTIVE</div>
                <div style="font-size:1.1rem;font-weight:700;margin-top:4px">{kz['active']}</div>
                <div style="color:#667eea;font-size:0.8rem">⏰ {kz['current_time']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(30,58,95,0.2);border-left:4px solid #1e3a5f;
                        border-radius:8px;padding:12px 18px;margin:4px 0;">
                <div style="color:#667eea;font-size:0.8rem;letter-spacing:2px">KILL ZONE</div>
                <div style="color:#e0e6f0">Aucune session active</div>
                <div style="color:#667eea;font-size:0.8rem">⏰ {kz['current_time']}</div>
            </div>""", unsafe_allow_html=True)

        if kz["next"]:
            st.markdown(f"""
            <div class="killzone-next">
                <div class="killzone-label-next">⏳ PROCHAINE KILL ZONE</div>
                <div style="font-size:1.05rem;font-weight:700;margin-top:4px;color:#b388ff">{kz['next']}</div>
                <div style="color:#667eea;font-size:0.8rem">🕐 Ouvre à {kz['next_time']}</div>
            </div>""", unsafe_allow_html=True)

        # Toutes les KZ
        st.markdown("**Calendrier des sessions (UTC)**")
        now_h = datetime.now(UTC).hour
        for kz_name, (s, e) in KILL_ZONES.items():
            is_act = s <= now_h < e
            bg = "rgba(255,107,53,0.15)" if is_act else "rgba(20,30,50,0.5)"
            bd = "#ff6b35" if is_act else "#1e3a5f"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {bd};border-radius:6px;
                        padding:5px 12px;margin:3px 0;display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.85rem">{kz_name}</span>
                <span style="color:#667eea;font-size:0.8rem">{s:02d}:00 – {e:02d}:00</span>
                {"<span style='color:#ff6b35;font-size:0.75rem;font-weight:700'>● LIVE</span>" if is_act else ""}
            </div>""", unsafe_allow_html=True)

        # ── PRÉDICTION ML
        st.markdown('<div class="section-title">🤖 PRÉDICTION ML</div>', unsafe_allow_html=True)

        with st.spinner("🧠 Entraînement du modèle ML..."):
            model, feat_cols, accuracy = train_ml(df)

        pred, proba = predict_signal(model, df, feat_cols)

        if pred is not None:
            signal_label = "🟢 ACHAT / HAUSSE" if pred == 1 else "🔴 VENTE / BAISSE"
            conf = proba[pred] * 100
            conf_color = "#00ff88" if conf > 65 else ("#ffd700" if conf > 50 else "#ff4d6d")
            sig_html_cls = "signal-buy" if pred == 1 else "signal-sell"

            # ── FENÊTRE PRÉDICTION ANIMÉE
            st.markdown(f"""
            <div class="pred-window">
                <div class="pred-title">🔮 ANALYSE PRÉDICTIVE EN COURS...</div>
                <div style="text-align:center;margin:18px 0;">
                    <span class="{sig_html_cls}" style="font-size:1.1rem;padding:10px 28px">{signal_label}</span>
                </div>
                <div class="metric-row"><span class="metric-key">Confiance ML</span>
                    <span class="metric-val" style="color:{conf_color}">{conf:.1f}%</span></div>
                <div class="metric-row"><span class="metric-key">Précision modèle</span>
                    <span class="metric-val">{accuracy}%</span></div>
                <div class="metric-row"><span class="metric-key">Z-Score</span>
                    <span class="metric-val">{last['Z_score']:.3f}</span></div>
                <div class="metric-row"><span class="metric-key">RSI</span>
                    <span class="metric-val">{last['RSI']:.1f}</span></div>
                <div class="metric-row"><span class="metric-key">BB Position</span>
                    <span class="metric-val">{last['BB_pos']*100:.1f}%</span></div>
                <div class="metric-row"><span class="metric-key">MACD Hist</span>
                    <span class="metric-val">{last['MACD_hist']:.4f}</span></div>
                <div class="metric-row"><span class="metric-key">Stochastique K</span>
                    <span class="metric-val">{last['Stoch_K']:.1f}</span></div>
                <div class="metric-row"><span class="metric-key">ATR</span>
                    <span class="metric-val">{last['ATR']:.4f}</span></div>
                <div class="metric-row"><span class="metric-key">EMA50 vs EMA200</span>
                    <span class="metric-val">{'✅ Haussier' if last['EMA50']>last['EMA200'] else '🔻 Baissier'}</span></div>
                <div class="metric-row"><span class="metric-key">Momentum (5p)</span>
                    <span class="metric-val">{last['Momentum']:.3f}%</span></div>
                <div style="margin-top:12px;background:#0a0e1a;border-radius:6px;height:10px;overflow:hidden;">
                    <div style="width:{conf:.0f}%;background:linear-gradient(90deg,#7b2ff7,{conf_color});
                                height:100%;border-radius:6px;transition:width 1s ease;"></div>
                </div>
                <div style="text-align:right;font-size:0.75rem;color:#667eea;margin-top:4px">Confiance: {conf:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            # ── JAUGE CONFIANCE
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                title={"text": "Confiance (%)", "font": {"color":"#e0e6f0", "family":"Orbitron", "size":12}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor":"#667eea"},
                    "bar":  {"color": conf_color},
                    "bgcolor": "#111827",
                    "bordercolor": "#1e3a5f",
                    "steps": [
                        {"range": [0,  50], "color": "rgba(255,77,109,0.15)"},
                        {"range": [50, 65], "color": "rgba(255,215,0,0.15)"},
                        {"range": [65,100], "color": "rgba(0,255,136,0.15)"},
                    ],
                    "threshold": {"line":{"color":"white","width":2},"value":conf}
                },
                number={"suffix":"%","font":{"color":conf_color,"family":"Orbitron","size":24}}
            ))
            gauge.update_layout(
                paper_bgcolor="#0d1526", font_color="#e0e6f0",
                height=200, margin=dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(gauge, use_container_width=True)

        else:
            st.warning("⚠️ Données insuffisantes pour la prédiction ML")

    # ── INDICATEURS SUPPLEMENTAIRES
    st.markdown('<div class="section-title">📊 INDICATEURS DÉTAILLÉS</div>', unsafe_allow_html=True)
    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
    indicators = [
        ("RSI (14)", f"{last['RSI']:.1f}", "Survente <30 / Surachat >70",
         "#ff4d6d" if last["RSI"]>70 else ("#00ff88" if last["RSI"]<30 else "#00d4ff")),
        ("MACD", f"{last['MACD']:.4f}", f"Hist: {last['MACD_hist']:+.4f}",
         "#00ff88" if last["MACD_hist"]>0 else "#ff4d6d"),
        ("Stoch K", f"{last['Stoch_K']:.1f}", f"D: {last['Stoch_D']:.1f}",
         "#ff4d6d" if last["Stoch_K"]>80 else ("#00ff88" if last["Stoch_K"]<20 else "#00d4ff")),
        ("ATR (14)", f"{last['ATR']:.4f}", "Volatilité", "#ffd700"),
        ("Momentum", f"{last['Momentum']:+.3f}%", "Sur 5 bougies",
         "#00ff88" if last["Momentum"]>0 else "#ff4d6d"),
    ]
    for col, (name, val, sub, color) in zip([ic1,ic2,ic3,ic4,ic5], indicators):
        with col:
            st.markdown(f"""
            <div class="price-card" style="text-align:center;">
                <div class="price-label">{name}</div>
                <div style="font-family:Orbitron;font-size:1.3rem;font-weight:700;color:{color}">{val}</div>
                <div style="color:#667eea;font-size:0.75rem">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── PIED DE PAGE
    st.markdown("---")
    ft1, ft2, ft3 = st.columns(3)
    with ft1:
        st.markdown(f"<span style='color:#667eea;font-size:0.8rem'>📡 Source: Yahoo Finance | Actif: **{asset_label}**</span>", unsafe_allow_html=True)
    with ft2:
        st.markdown(f"<span style='color:#667eea;font-size:0.8rem;text-align:center'>🔄 TF: {tf_label} | BB({bb_window},{bb_std}) | Z({z_window})</span>", unsafe_allow_html=True)
    with ft3:
        st.markdown(f"<span style='color:#445;font-size:0.75rem'>⚠️ Outil éducatif uniquement — Pas un conseil financier</span>", unsafe_allow_html=True)

    # ── AUTO-REFRESH
    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
