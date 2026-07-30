"""Interactive test site for the black box: pick a symbol (or upload
your own OHLCV data), tune the strategy/risk parameters in the
sidebar, and see exactly what the engine would do -- the three-tier
signal fallback, the ATR stop/take-profit it would set, and a full
cost-aware backtest -- without touching a broker.

Run with:
    pip install -r requirements-web.txt
    streamlit run webapp/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from blackbox.alpha.features import average_true_range, build_feature_matrix
from blackbox.alpha.labeling import apply_triple_barrier, meta_label
from blackbox.alpha.model import MetaLabelingModel
from blackbox.alpha.signals import MeanReversionSignal, MomentumSignal, PriceActionConfluenceSignal
from blackbox.backtest.backtester import run_backtest
from blackbox.data.bars import get_daily_vol
from blackbox.data.market_data import DataCleaner, YFinanceProvider
from blackbox.risk.risk_manager import RiskLimits, RiskManager

st.set_page_config(page_title="Black-Box - Site de test", layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _generate_synthetic_ohlcv(n: int, regime: str, seed: int) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)

    if regime == "Mean-reverting (Ornstein-Uhlenbeck)":
        theta, mu, sigma = 0.15, 100.0, 0.5
        close = np.full(n, mu)
        for i in range(1, n):
            close[i] = close[i - 1] + theta * (mu - close[i - 1]) + sigma * rng.normal()
    elif regime == "Tendance":
        close = 100 + np.arange(n) * 0.05 + rng.normal(0, 0.5, n)
    else:  # random walk
        close = 100 + np.cumsum(rng.normal(0, 1, n))

    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.05, n),
            "high": close + np.abs(rng.normal(0, 0.2, n)),
            "low": close - np.abs(rng.normal(0, 0.2, n)),
            "close": close,
            "volume": rng.integers(1000, 10_000, n),
        },
        index=idx,
    )
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


def _load_csv(uploaded_file) -> pd.DataFrame:
    raw = pd.read_csv(uploaded_file)
    date_col = next((c for c in raw.columns if c.lower() in ("date", "timestamp", "time")), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col], utc=True)
    raw = raw.set_index(date_col).sort_index()
    raw.columns = [c.lower() for c in raw.columns]
    missing = {"open", "high", "low", "close", "volume"} - set(raw.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")
    return raw[["open", "high", "low", "close", "volume"]]


@st.cache_data(show_spinner=False)
def _fetch_yfinance(symbol: str, start: str, bar_size: str) -> pd.DataFrame:
    provider = YFinanceProvider()
    return provider.get_historical_bars(symbol, start=start, bar_size=bar_size)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Source de données")
source = st.sidebar.radio(
    "Origine des données",
    ["Données synthétiques (démo)", "Données réelles (yfinance)", "Importer un CSV"],
    help="Les données synthétiques marchent hors-ligne ; yfinance nécessite un accès réseau réel.",
)

df: pd.DataFrame | None = None
load_error: str | None = None

if source == "Données synthétiques (démo)":
    regime = st.sidebar.selectbox(
        "Régime simulé", ["Mean-reverting (Ornstein-Uhlenbeck)", "Tendance", "Marche aléatoire"]
    )
    n_bars = st.sidebar.slider("Nombre de bougies", 200, 3000, 800, step=100)
    seed = st.sidebar.number_input("Seed aléatoire", value=42, step=1)
    df = _generate_synthetic_ohlcv(n_bars, regime, int(seed))
    label = f"synthétique ({regime}, {n_bars} bougies)"

elif source == "Données réelles (yfinance)":
    symbol = st.sidebar.text_input("Symbole", value="SPY")
    bar_size = st.sidebar.selectbox("Taille de bougie", ["1D", "1H", "1min"])
    start = st.sidebar.text_input("Début", value="500d ago")
    if st.sidebar.button("Charger"):
        try:
            df = _fetch_yfinance(symbol, start, bar_size)
        except Exception as exc:
            load_error = str(exc)
    label = f"{symbol} ({bar_size}, yfinance)"

else:
    uploaded = st.sidebar.file_uploader("Fichier CSV (colonnes: date, open, high, low, close, volume)")
    if uploaded is not None:
        try:
            df = _load_csv(uploaded)
        except Exception as exc:
            load_error = str(exc)
    label = "CSV importé"

st.sidebar.header("Paramètres de stratégie")
lookback = st.sidebar.slider("Lookback mean-reversion", 20, 120, 60)
entry_z = st.sidebar.slider("Z-score d'entrée", 0.5, 3.0, 2.0)
exit_z = st.sidebar.slider("Z-score de sortie", 0.1, 1.5, 0.5)
fast = st.sidebar.slider("MA rapide (momentum)", 5, 50, 20)
slow = st.sidebar.slider("MA lente (momentum)", 50, 200, 100)

st.sidebar.header("Risque & coûts")
capital = st.sidebar.number_input("Capital", value=100_000.0, step=10_000.0)
commission_bps = st.sidebar.number_input("Commission (bps)", value=0.5)
slippage_bps = st.sidebar.number_input("Slippage (bps)", value=1.0)
atr_stop_mult = st.sidebar.number_input("Stop (x ATR)", value=2.5)
atr_tp_mult = st.sidebar.number_input("Take-profit (x ATR)", value=5.0)

train_meta = st.sidebar.checkbox(
    "Entraîner le meta-modèle ML (Random Forest + PurgedKFold)",
    help="Plus lent : entraîne le classifieur de meta-labeling sur l'historique chargé.",
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("📈 Black-Box — Site de test interactif")
st.caption(
    "Backtest et inspection des signaux (mean-reversion, momentum, confluence price-action) "
    "sur des données réelles ou synthétiques, sans jamais toucher un broker."
)

if load_error:
    st.error(f"Erreur de chargement des données : {load_error}")
    st.stop()

if df is None or df.empty:
    st.info("Choisis une source de données dans la barre latérale pour commencer.")
    st.stop()

df = DataCleaner.clean(df)
if len(df) < max(slow, lookback) + 10:
    st.warning("Pas assez de bougies pour ces paramètres de lookback -- réduis-les ou charge plus d'historique.")
    st.stop()

st.success(f"Données chargées : {label} — {len(df)} bougies, du {df.index[0].date()} au {df.index[-1].date()}")

tab_backtest, tab_signals, tab_data = st.tabs(["🧪 Backtest", "🔍 Signaux", "🗂 Données brutes"])

# -- Backtest tab -------------------------------------------------------------

with tab_backtest:
    result = run_backtest(
        df,
        starting_capital=capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        lookback=lookback,
    )

    cols = st.columns(5)
    cols[0].metric("Sharpe", f"{result.sharpe:.2f}")
    cols[1].metric("Sortino", f"{result.sortino:.2f}")
    cols[2].metric("Max Drawdown", f"{result.max_drawdown * 100:.1f}%")
    cols[3].metric("CAGR", f"{result.cagr * 100:.1f}%")
    cols[4].metric("Turnover moyen", f"{result.turnover:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve, name="Équity", line=dict(color="#4C9AFF")))
    fig.update_layout(title="Courbe d'équity (coûts inclus)", xaxis_title="Date", yaxis_title="Équity ($)", height=420)
    st.plotly_chart(fig, use_container_width=True)

# -- Signals tab -------------------------------------------------------------

with tab_signals:
    mr = MeanReversionSignal(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
    mom = MomentumSignal(fast=fast, slow=slow)

    # Pick a confirmation timeframe strictly coarser than the data's own
    # cadence (same rule the engine applies) -- resampling to something
    # finer than the source bars would silently produce a mostly-NaN
    # trend filter.
    median_delta = df.index.to_series().diff().median()
    if median_delta <= pd.Timedelta(minutes=1):
        htf_rule = "15min"
    elif median_delta <= pd.Timedelta(minutes=15):
        htf_rule = "1h"
    elif median_delta <= pd.Timedelta(hours=1):
        htf_rule = "4h"
    elif median_delta <= pd.Timedelta(hours=4):
        htf_rule = "1D"
    else:
        htf_rule = "1W"
    pa = PriceActionConfluenceSignal(htf_rule=htf_rule)

    with st.spinner("Calcul des signaux (le test ADF de mean-reversion peut prendre quelques secondes)..."):
        mr_result = mr.generate(df)
        mom_result = mom.generate(df)
        pa_result = pa.generate(df)

    last_mr, last_mom, last_pa = int(mr_result.side.iloc[-1]), int(mom_result.side.iloc[-1]), int(pa_result.side.iloc[-1])
    if last_mr != 0:
        active_tier, primary_side = "Mean-reversion", last_mr
    elif last_mom != 0:
        active_tier, primary_side = "Momentum", last_mom
    else:
        active_tier, primary_side = "Price-action (confluence)", last_pa

    direction = {1: "LONG 🟢", -1: "SHORT 🔴", 0: "FLAT ⚪"}[primary_side]
    st.subheader(f"Décision actuelle : {direction}")
    st.caption(f"Régime actif : **{active_tier}** (ordre de repli mean-reversion → momentum → price-action)")

    atr = float(average_true_range(df).iloc[-1])
    price = float(df["close"].iloc[-1])
    if primary_side != 0:
        limits = RiskLimits(atr_stop_multiple=atr_stop_mult, atr_tp_multiple=atr_tp_mult)
        rm = RiskManager(limits=limits, starting_equity=capital)
        stop = rm.compute_stop_price(price, atr, primary_side)
        tp = rm.compute_take_profit_price(price, atr, primary_side)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix actuel", f"{price:.4f}")
        c2.metric("Stop-loss (ATR)", f"{stop:.4f}")
        c3.metric("Take-profit (ATR)", f"{tp:.4f}")

    if train_meta:
        with st.spinner("Entraînement du meta-modèle (PurgedKFold)..."):
            primary = mr_result.side[mr_result.side != 0]
            if len(primary) < 100:
                st.warning("Pas assez de signaux mean-reversion pour entraîner le meta-modèle sur cet historique.")
            else:
                daily_vol = get_daily_vol(df["close"])
                barriers = apply_triple_barrier(df["close"], primary.index, daily_vol)
                labels = meta_label(primary, barriers)
                features = build_feature_matrix(df, lookback=lookback)
                aligned = features.reindex(labels.index).dropna()
                labels = labels.reindex(aligned.index)
                if len(aligned) < 50:
                    st.warning("Pas assez d'échantillons alignés pour entraîner le meta-modèle.")
                else:
                    model = MetaLabelingModel()
                    label_end_times = barriers.reindex(aligned.index)["t1"]
                    meta_result = model.fit_with_purged_cv(aligned, labels, label_end_times)
                    st.metric("Précision OOS (PurgedKFold)", f"{meta_result.oos_accuracy * 100:.1f}%")
                    st.bar_chart(meta_result.feature_importances)

    st.subheader("Prix et points d'entrée dans le temps")
    st.caption(
        "Seuls les changements de régime (entrées) sont marqués -- le momentum est "
        "presque toujours \"dans le marché\" et marquer chaque bougie noierait le graphique."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="#888")))

    for name, res, color, symbol_marker in (
        ("Mean-reversion", mr_result, "#2ecc71", "circle"),
        ("Momentum", mom_result, "#3498db", "triangle-up"),
        ("Price-action", pa_result, "#e67e22", "star"),
    ):
        side = res.side.fillna(0)
        entered = side.diff().fillna(side)  # first non-zero value also counts as an entry
        long_entries = df.index[(side == 1) & (entered != 0)]
        short_entries = df.index[(side == -1) & (entered != 0)]
        if len(long_entries):
            fig.add_trace(go.Scatter(
                x=long_entries, y=df.loc[long_entries, "close"], mode="markers", name=f"{name} LONG",
                marker=dict(color=color, symbol=symbol_marker, size=11, line=dict(width=1, color="black")),
            ))
        if len(short_entries):
            fig.add_trace(go.Scatter(
                x=short_entries, y=df.loc[short_entries, "close"], mode="markers", name=f"{name} SHORT",
                marker=dict(color=color, symbol=symbol_marker, size=11, line=dict(width=2, color="red")),
            ))

    fig.update_layout(height=520, xaxis_title="Date", yaxis_title="Prix")
    st.plotly_chart(fig, use_container_width=True)

# -- Raw data tab -------------------------------------------------------------

with tab_data:
    st.dataframe(df.tail(300), use_container_width=True)
    st.write(df.describe())
