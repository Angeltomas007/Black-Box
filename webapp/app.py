"""Black-Box Scanner: type a ticker, pick a timeframe, and get a clear
recommendation card (buy / sell / flat, with an ATR stop/take-profit
and a suggested holding horizon) built from the same three-tier signal
fallback the live engine uses -- mean-reversion, then momentum, then
price-action confluence.

Advanced controls (strategy/risk parameters, offline synthetic data,
CSV import, the detailed backtest, and the ML meta-model) live in
collapsed sections below the scanner so the default experience stays a
one-line search.

Run with:
    pip install -r requirements-web.txt
    streamlit run webapp/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from blackbox.alpha.features import average_true_range, build_feature_matrix, rsi as compute_rsi
from blackbox.alpha.labeling import apply_triple_barrier, meta_label
from blackbox.alpha.model import MetaLabelingModel
from blackbox.alpha.signals import MeanReversionSignal, MomentumSignal, PriceActionConfluenceSignal
from blackbox.backtest.backtester import run_backtest
from blackbox.data.bars import get_daily_vol
from blackbox.data.market_data import DataCleaner, YFinanceProvider
from blackbox.risk.risk_manager import RiskLimits, RiskManager

st.set_page_config(page_title="Black-Box Scanner", layout="wide", page_icon="📈")

# ---------------------------------------------------------------------------
# Style -- validated palette (dataviz skill): status colors for the
# recommendation badge, categorical slots 1/2/3 (blue/orange/aqua) for
# the three signal tiers in the chart, both cleared for CVD-safety as a
# set. Surfaces/ink follow the light/dark reference pairs verbatim.
# ---------------------------------------------------------------------------

GOOD, CRITICAL, MUTED = "#0ca30c", "#d03b3b", "#898781"
TIER_COLORS = {"Mean-reversion": "#2a78d6", "Momentum": "#eb6834", "Price-action": "#1baf7a"}
AUTO_REFRESH_SECONDS = 30

st.markdown(
    """
    <style>
    :root {
        --bb-surface: #fcfcfb; --bb-text: #0b0b0b;
        --bb-text-secondary: #52514e; --bb-border: rgba(11,11,11,0.10);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --bb-surface: #1a1a19; --bb-text: #ffffff;
            --bb-text-secondary: #c3c2b7; --bb-border: rgba(255,255,255,0.10);
        }
    }
    .bb-card {
        background: var(--bb-surface); border: 1px solid var(--bb-border);
        border-radius: 16px; padding: 24px 30px; margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .bb-symbol { font-size: 1.05rem; color: var(--bb-text-secondary); font-weight: 600; letter-spacing: 0.03em; }
    .bb-price { font-size: 2.3rem; font-weight: 700; color: var(--bb-text); font-variant-numeric: tabular-nums; }
    .bb-badge {
        display: inline-flex; align-items: center; gap: 8px; font-size: 1.3rem;
        font-weight: 700; padding: 6px 20px; border-radius: 999px; color: #fff;
    }
    .bb-sub { color: var(--bb-text-secondary); font-size: 0.95rem; margin-top: 8px; }
    .bb-tier {
        text-align: center; padding: 14px 10px; border-radius: 12px;
        border: 1px solid var(--bb-border); background: var(--bb-surface);
    }
    .bb-tier-name { color: var(--bb-text-secondary); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
    .bb-tier-val { font-size: 1.15rem; font-weight: 700; margin-top: 4px; color: var(--bb-text); }
    .bb-section-title { margin-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Timeframe presets -- entry timeframe + a strictly coarser confirmation
# timeframe, mirroring blackbox.core.engine's _HTF_TREND_MAP mapping so
# the scanner matches what the live engine would actually compute.
# ---------------------------------------------------------------------------

TIMEFRAME_OPTIONS: dict[str, dict[str, str]] = {
    "⚡ 5 min (scalp)": dict(bar_size="5min", htf_rule="30min", start="10d ago", horizon="15 à 30 minutes (~3-6 bougies)"),
    "🔹 15 min (intraday)": dict(bar_size="15min", htf_rule="1h", start="30d ago", horizon="45 à 90 minutes (~3-6 bougies)"),
    "🔸 1 heure (swing court)": dict(bar_size="1H", htf_rule="1D", start="90d ago", horizon="3 à 6 heures"),
    "📅 1 jour (position)": dict(bar_size="1D", htf_rule="1W", start="500d ago", horizon="3 à 6 jours"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _generate_synthetic_ohlcv(n: int, freq: str, regime: str, seed: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)

    if regime == "mean_reverting":
        theta, mu, sigma = 0.15, 100.0, 0.5
        close = np.full(n, mu)
        for i in range(1, n):
            close[i] = close[i - 1] + theta * (mu - close[i - 1]) + sigma * rng.normal()
    elif regime == "trending":
        close = 100 + np.arange(n) * 0.05 + rng.normal(0, 0.5, n)
    else:
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


@st.cache_data(show_spinner=False, ttl=20)
def _fetch_yfinance(symbol: str, start: str, bar_size: str) -> pd.DataFrame:
    provider = YFinanceProvider()
    return provider.get_historical_bars(symbol, start=start, bar_size=bar_size)


@st.cache_data(show_spinner=False, ttl=300)
def _search_symbols(query: str) -> list[dict]:
    """Resolve a company name (or partial ticker) to candidate symbols.

    Names don't always resemble their ticker (Broadcom -> AVGO), so
    forcing users to already know the ticker is a real usability gap.
    Uses yfinance's own Search (the same client/anti-blocking layer as
    the price fetch above) rather than a raw call to Yahoo's endpoint,
    so it works wherever the price fetch already works.
    """
    query = query.strip()
    if len(query) < 2:
        return []
    import yfinance as yf

    try:
        quotes = yf.Search(query, max_results=8).quotes
    except Exception:
        return []

    results = []
    for q in quotes:
        sym = q.get("symbol")
        if not sym:
            continue
        results.append(
            {
                "symbol": sym,
                "name": q.get("shortname") or q.get("longname") or sym,
                "exchange": q.get("exchange", ""),
                "type": q.get("quoteType", ""),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Header + scanner search bar (static -- lives outside the auto-refreshing
# fragment since changing symbol/timeframe/params should always trigger a
# full rerun, not wait for the next refresh tick).
# ---------------------------------------------------------------------------

st.title("📈 Black-Box Scanner")
st.caption(
    "Cherche une action, l'analyse tourne sur la bougie d'entrée choisie (par défaut 5 min) "
    "confirmée par une tendance de plus haut niveau -- exactement le fallback "
    "mean-reversion → momentum → price-action du moteur live."
)

col_symbol, col_tf, col_btn, col_demo, col_refresh = st.columns([2.6, 2.2, 1.2, 1.5, 1.5])
with col_symbol:
    search_query = st.text_input(
        "Rechercher une action",
        value="Oracle",
        placeholder="ex : Apple, Oracle, Broadcom, ou directement AAPL",
        label_visibility="collapsed",
    )
with col_tf:
    tf_label = st.selectbox("Horizon d'analyse", list(TIMEFRAME_OPTIONS.keys()), label_visibility="collapsed")
with col_btn:
    st.button("🔍 Analyser", use_container_width=True, type="primary")
with col_demo:
    demo_mode = st.checkbox("Mode démo (hors-ligne)", help="Données synthétiques -- utile si le marché est fermé ou le réseau indisponible.")
with col_refresh:
    auto_refresh = st.checkbox(
        f"🔄 Auto ({AUTO_REFRESH_SECONDS}s)",
        help="Recharge les données et relance l'analyse toutes les 30 secondes, sans re-cliquer sur Analyser.",
    )

# Resolve the free-text query to a real ticker -- by name (company
# search) if we get a confident match, falling back to treating the
# text as a literal ticker (so "AAPL" typed directly still works even
# offline in demo mode, where no search call is made).
matched_name: str | None = None
if demo_mode:
    symbol = search_query.strip().upper()
else:
    matches = _search_symbols(search_query)
    if matches:
        options = [
            f"{m['symbol']} — {m['name']}" + (f" ({m['exchange']})" if m["exchange"] else "")
            for m in matches
        ]
        col_match, col_match_info = st.columns([2.6, 4.7])
        with col_match:
            chosen = st.selectbox("Résultat", options, label_visibility="collapsed", key="symbol_match")
        picked = matches[options.index(chosen)]
        symbol, matched_name = picked["symbol"], picked["name"]
        with col_match_info:
            st.caption(f"Symbole utilisé : **{symbol}** — {matched_name}")
    else:
        symbol = search_query.strip().upper()
        if search_query.strip():
            st.caption(
                f"Aucune correspondance trouvée pour « {search_query} » -- utilisé tel quel comme symbole : **{symbol}**"
            )

tf_conf = TIMEFRAME_OPTIONS[tf_label]

with st.expander("⚙️ Paramètres avancés (stratégie, risque, données)"):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        st.markdown("**Mean-reversion**")
        lookback = st.slider("Lookback", 20, 120, 60)
        entry_z = st.slider("Z-score d'entrée", 0.5, 3.0, 2.0)
        exit_z = st.slider("Z-score de sortie", 0.1, 1.5, 0.5)
    with adv_col2:
        st.markdown("**Momentum**")
        fast = st.slider("MA rapide", 5, 50, 20)
        slow = st.slider("MA lente", 50, 200, 100)
        train_meta = st.checkbox("Entraîner le meta-modèle ML (Random Forest + PurgedKFold)")
    with adv_col3:
        st.markdown("**Risque & coûts**")
        capital = st.number_input("Capital", value=100_000.0, step=10_000.0)
        commission_bps = st.number_input("Commission (bps)", value=0.5)
        slippage_bps = st.number_input("Slippage (bps)", value=1.0)
        atr_stop_mult = st.number_input("Stop (x ATR)", value=2.5)
        atr_tp_mult = st.number_input("Take-profit (x ATR)", value=5.0)

    st.markdown("**Import CSV** (remplace la recherche par action pour ce run)")
    uploaded = st.file_uploader("Colonnes attendues : date, open, high, low, close, volume", label_visibility="collapsed")

st.divider()

# ---------------------------------------------------------------------------
# Everything below reruns on its own every AUTO_REFRESH_SECONDS when the
# checkbox is on -- st.fragment isolates that rerun to this block instead
# of re-executing the whole script (and re-running st.stop() inside it
# only halts the fragment, not the page).
# ---------------------------------------------------------------------------

@st.fragment(run_every=AUTO_REFRESH_SECONDS if auto_refresh else None)
def render_scanner() -> None:
    df: pd.DataFrame | None = None
    load_error: str | None = None
    label = ""

    if uploaded is not None:
        try:
            df = _load_csv(uploaded)
            label = "CSV importé"
        except Exception as exc:
            load_error = str(exc)
    elif demo_mode:
        freq = {"5min": "5min", "15min": "15min", "1H": "1h", "1D": "1D"}[tf_conf["bar_size"]]
        n_bars = 600 if tf_conf["bar_size"] in ("5min", "15min") else 400
        df = _generate_synthetic_ohlcv(n_bars, freq, "mean_reverting", seed=hash(symbol) % 1000)
        label = f"{symbol.upper()} (démo synthétique, {tf_label})"
    else:
        try:
            df = _fetch_yfinance(symbol.strip().upper(), tf_conf["start"], tf_conf["bar_size"])
            label = f"{symbol.upper()} ({tf_label}, yfinance)"
        except Exception as exc:
            load_error = str(exc)

    if load_error:
        st.error(
            f"Impossible de charger **{symbol.upper()}** en direct : {load_error}\n\n"
            "Coche **Mode démo (hors-ligne)** pour tester le scanner sans accès réseau, "
            "ou vérifie le symbole."
        )
        st.stop()

    if df is None or df.empty:
        st.info("Entre un symbole et clique sur **Analyser** pour lancer le scan.")
        st.stop()

    df = DataCleaner.clean(df)
    min_required = max(slow, lookback) + 10
    if len(df) < min_required:
        st.warning(
            f"Pas assez de bougies ({len(df)} chargées, {min_required} nécessaires) pour ces paramètres -- "
            "élargis la fenêtre de données ou réduis les lookbacks dans les paramètres avancés."
        )
        st.stop()

    # -- Three-tier signal fallback (mean-reversion -> momentum -> price-action) --

    mr = MeanReversionSignal(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
    mom = MomentumSignal(fast=fast, slow=slow)
    pa = PriceActionConfluenceSignal(htf_rule=tf_conf["htf_rule"])

    with st.spinner("Analyse en cours (le test ADF de stationnarité peut prendre quelques secondes)..."):
        mr_result = mr.generate(df)
        mom_result = mom.generate(df)
        pa_result = pa.generate(df)

    last_mr = int(mr_result.side.iloc[-1])
    last_mom = int(mom_result.side.iloc[-1])
    last_pa = int(pa_result.side.iloc[-1])
    pa_strength = float(pa_result.strength.iloc[-1])

    if last_mr != 0:
        active_tier, primary_side = "Mean-reversion", last_mr
    elif last_mom != 0:
        active_tier, primary_side = "Momentum", last_mom
    else:
        active_tier, primary_side = "Price-action", last_pa

    confidence = pa_strength if active_tier == "Price-action" and primary_side != 0 else 0.5

    price = float(df["close"].iloc[-1])
    atr = float(average_true_range(df).iloc[-1])
    last_ts = df.index[-1]

    # -- Recommendation card --

    badge_color = {1: GOOD, -1: CRITICAL, 0: MUTED}[primary_side]
    badge_text = {1: "🟢 ACHAT", -1: "🔴 VENTE À DÉCOUVERT", 0: "⚪ NEUTRE"}[primary_side]
    header_name = f"{symbol.upper()} — {matched_name}" if matched_name else symbol.upper()

    st.markdown(
        f"""
        <div class="bb-card" style="border-left: 5px solid {badge_color};">
            <div class="bb-symbol">{header_name} · {last_ts.strftime('%Y-%m-%d %H:%M UTC')}</div>
            <div style="display:flex; align-items:baseline; gap:18px; margin:6px 0;">
                <span class="bb-price">{price:,.2f}</span>
                <span class="bb-badge" style="background:{badge_color};">{badge_text}</span>
            </div>
            <div class="bb-sub">
                Régime actif : <b>{active_tier}</b> &nbsp;·&nbsp;
                Horizon indicatif : <b>{tf_conf['horizon']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if auto_refresh:
        st.caption(f"🔄 Dernière actualisation : {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} (auto, toutes les {AUTO_REFRESH_SECONDS}s)")

    card_col, gauge_col = st.columns([2, 1])

    with card_col:
        if primary_side != 0:
            limits = RiskLimits(atr_stop_multiple=atr_stop_mult, atr_tp_multiple=atr_tp_mult)
            rm = RiskManager(limits=limits, starting_equity=capital)
            stop = rm.compute_stop_price(price, atr, primary_side)
            tp = rm.compute_take_profit_price(price, atr, primary_side)
            risk = abs(price - stop)
            reward = abs(tp - price)
            rr = reward / risk if risk > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Entrée", f"{price:.4f}")
            m2.metric("Stop-loss (ATR)", f"{stop:.4f}", delta=f"{(stop / price - 1) * 100:.2f}%")
            m3.metric("Take-profit (ATR)", f"{tp:.4f}", delta=f"{(tp / price - 1) * 100:.2f}%")
            m4.metric("Risque/Rendement", f"1:{rr:.2f}")
        else:
            st.info("Aucun signal actif : les trois régimes (mean-reversion, momentum, price-action) sont neutres en ce moment.")

        st.markdown("**Pourquoi cette décision ?**")
        t1, t2, t3 = st.columns(3)
        for col, name, side in (
            (t1, "Mean-reversion", last_mr),
            (t2, "Momentum", last_mom),
            (t3, "Price-action", last_pa),
        ):
            icon = "🟢" if side == 1 else "🔴" if side == -1 else "⚪"
            word = "LONG" if side == 1 else "SHORT" if side == -1 else "Neutre"
            col.markdown(
                f'<div class="bb-tier"><div class="bb-tier-name">{name}</div>'
                f'<div class="bb-tier-val">{icon} {word}</div></div>',
                unsafe_allow_html=True,
            )

    with gauge_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "Confiance", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": badge_color},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(137,135,129,0.18)"},
                    {"range": [40, 70], "color": "rgba(137,135,129,0.30)"},
                    {"range": [70, 100], "color": "rgba(137,135,129,0.42)"},
                ],
            },
        ))
        gauge.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True, key="confidence_gauge")
        st.caption(
            "50% = régime neutre par défaut (pas de meta-modèle entraîné). "
            "Coche l'entraînement du meta-modèle dans les paramètres avancés pour une "
            "confiance calibrée par validation croisée purgée."
        )

    st.success(f"Données : {label} — {len(df)} bougies, du {df.index[0]} au {df.index[-1]}")

    # -- Candlestick chart with RSI panel and entry markers --

    st.subheader("Bougies, RSI et points d'entrée")
    st.caption(
        "Fenêtre récente uniquement -- seuls les changements de régime (entrées) sont marqués, "
        "la couleur identifie le signal, la forme la direction (▲ long / ▼ short). Lignes "
        "pointillées RSI : survente (35) / surachat (65), seuils utilisés par le régime price-action."
    )

    window = df.tail(min(300, len(df)))
    rsi_series = compute_rsi(df["close"]).reindex(window.index)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=window.index, open=window["open"], high=window["high"], low=window["low"], close=window["close"],
            name="Prix", increasing_line_color=GOOD, decreasing_line_color=CRITICAL,
            increasing_fillcolor=GOOD, decreasing_fillcolor=CRITICAL,
        ),
        row=1, col=1,
    )

    for name, res in (("Mean-reversion", mr_result), ("Momentum", mom_result), ("Price-action", pa_result)):
        side = res.side.reindex(window.index).fillna(0)
        entered = side.diff().fillna(side)
        longs = window.index[(side == 1) & (entered != 0)]
        shorts = window.index[(side == -1) & (entered != 0)]
        color = TIER_COLORS[name]
        if len(longs):
            fig.add_trace(
                go.Scatter(
                    x=longs, y=window.loc[longs, "low"] * 0.995, mode="markers", name=f"{name} ▲",
                    marker=dict(color=color, symbol="triangle-up", size=12, line=dict(width=1, color="rgba(0,0,0,0.3)")),
                ),
                row=1, col=1,
            )
        if len(shorts):
            fig.add_trace(
                go.Scatter(
                    x=shorts, y=window.loc[shorts, "high"] * 1.005, mode="markers", name=f"{name} ▼",
                    marker=dict(color=color, symbol="triangle-down", size=12, line=dict(width=1, color="rgba(0,0,0,0.3)")),
                ),
                row=1, col=1,
            )

    fig.add_trace(
        go.Scatter(x=window.index, y=rsi_series, name="RSI", line=dict(color=TIER_COLORS["Price-action"], width=1.5)),
        row=2, col=1,
    )
    fig.add_hline(y=65, line_dash="dot", line_color=CRITICAL, opacity=0.6, row=2, col=1)
    fig.add_hline(y=35, line_dash="dot", line_color=GOOD, opacity=0.6, row=2, col=1)

    fig.update_layout(
        height=620, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, key="main_chart")

    # -- Advanced sections: meta-model, full backtest, raw data --

    if train_meta:
        with st.expander("🤖 Meta-modèle ML (Random Forest + PurgedKFold)", expanded=True):
            with st.spinner("Entraînement (validation croisée purgée)..."):
                primary = mr_result.side[mr_result.side != 0]
                if len(primary) < 100:
                    st.warning("Pas assez de signaux mean-reversion sur cet historique pour entraîner le meta-modèle.")
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

    with st.expander("🧪 Backtest détaillé"):
        result = run_backtest(
            df, starting_capital=capital, commission_bps=commission_bps,
            slippage_bps=slippage_bps, lookback=lookback,
        )
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Sharpe", f"{result.sharpe:.2f}")
        b2.metric("Sortino", f"{result.sortino:.2f}")
        b3.metric("Max Drawdown", f"{result.max_drawdown * 100:.1f}%")
        b4.metric("CAGR", f"{result.cagr * 100:.1f}%")
        b5.metric("Turnover moyen", f"{result.turnover:.3f}")

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve, name="Équity", line=dict(color=TIER_COLORS["Mean-reversion"])))
        eq_fig.update_layout(title="Courbe d'équity (coûts inclus)", template="plotly_white", height=380, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(eq_fig, use_container_width=True, key="equity_chart")

    with st.expander("🗂 Données brutes"):
        st.dataframe(df.tail(300), use_container_width=True)
        st.write(df.describe())


render_scanner()
