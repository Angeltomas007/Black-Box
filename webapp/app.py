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
    /* Reclaim the top space Streamlit reserves for its (now hidden,
       see .streamlit/config.toml) toolbar, so the page reads as a
       standalone site rather than an app shell with a dead header. */
    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px; }
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
    .bb-badge-sm {
        display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem;
        font-weight: 700; padding: 3px 12px; border-radius: 999px; color: #fff;
    }
    .bb-sub { color: var(--bb-text-secondary); font-size: 0.95rem; margin-top: 8px; }
    .bb-tier {
        text-align: center; padding: 14px 10px; border-radius: 12px;
        border: 1px solid var(--bb-border); background: var(--bb-surface);
    }
    .bb-tier-name { color: var(--bb-text-secondary); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
    .bb-tier-val { font-size: 1.15rem; font-weight: 700; margin-top: 4px; color: var(--bb-text); }
    .bb-mini-title { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
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


# Yahoo's search endpoint is scraped (undocumented) and occasionally
# rate-limits or blocks requests from hosted environments even while the
# price-history endpoint keeps working -- when that happens, a name search
# for "Oracle" would otherwise silently fall through to using the literal
# text "ORACLE" as a ticker, which doesn't exist. This static map covers
# the names people are most likely to type, so the scanner still resolves
# them correctly even if the live search is down.
_KNOWN_TICKERS: dict[str, tuple[str, str]] = {
    "apple": ("AAPL", "Apple Inc."),
    "oracle": ("ORCL", "Oracle Corporation"),
    "microsoft": ("MSFT", "Microsoft Corporation"),
    "amazon": ("AMZN", "Amazon.com Inc."),
    "google": ("GOOGL", "Alphabet Inc. (Google)"),
    "alphabet": ("GOOGL", "Alphabet Inc."),
    "meta": ("META", "Meta Platforms Inc."),
    "facebook": ("META", "Meta Platforms Inc."),
    "tesla": ("TSLA", "Tesla Inc."),
    "nvidia": ("NVDA", "NVIDIA Corporation"),
    "broadcom": ("AVGO", "Broadcom Inc."),
    "netflix": ("NFLX", "Netflix Inc."),
    "intel": ("INTC", "Intel Corporation"),
    "amd": ("AMD", "Advanced Micro Devices Inc."),
    "coca cola": ("KO", "Coca-Cola Company"),
    "coca-cola": ("KO", "Coca-Cola Company"),
    "mcdonald": ("MCD", "McDonald's Corporation"),
    "disney": ("DIS", "Walt Disney Company"),
    "walmart": ("WMT", "Walmart Inc."),
    "visa": ("V", "Visa Inc."),
    "mastercard": ("MA", "Mastercard Inc."),
    "jpmorgan": ("JPM", "JPMorgan Chase & Co."),
    "berkshire": ("BRK-B", "Berkshire Hathaway Inc."),
    "boeing": ("BA", "Boeing Company"),
    "ibm": ("IBM", "IBM"),
    "salesforce": ("CRM", "Salesforce Inc."),
    "adobe": ("ADBE", "Adobe Inc."),
    "paypal": ("PYPL", "PayPal Holdings Inc."),
    "uber": ("UBER", "Uber Technologies Inc."),
    "spotify": ("SPOT", "Spotify Technology S.A."),
    "airbnb": ("ABNB", "Airbnb Inc."),
    "starbucks": ("SBUX", "Starbucks Corporation"),
    "nike": ("NKE", "Nike Inc."),
    "pfizer": ("PFE", "Pfizer Inc."),
    "exxon": ("XOM", "Exxon Mobil Corporation"),
    "chevron": ("CVX", "Chevron Corporation"),
    "goldman sachs": ("GS", "Goldman Sachs Group Inc."),
    "ford": ("F", "Ford Motor Company"),
    "general motors": ("GM", "General Motors Company"),
    "qualcomm": ("QCOM", "Qualcomm Inc."),
    "shopify": ("SHOP", "Shopify Inc."),
    "square": ("SQ", "Block Inc."),
    "block": ("SQ", "Block Inc."),
    "palantir": ("PLTR", "Palantir Technologies Inc."),
}


@st.cache_data(show_spinner=False, ttl=300)
def _search_symbols(query: str) -> list[dict]:
    """Resolve a company name (or partial ticker) to candidate symbols.

    Names don't always resemble their ticker (Broadcom -> AVGO), so
    forcing users to already know the ticker is a real usability gap.
    Uses yfinance's own Search (the same client/anti-blocking layer as
    the price fetch above) rather than a raw call to Yahoo's endpoint,
    so it works wherever the price fetch already works. Falls back to a
    static name map (see _KNOWN_TICKERS) if the live search comes back
    empty or errors out.
    """
    query = query.strip()
    if len(query) < 2:
        return []

    results: list[dict] = []
    try:
        import yfinance as yf

        quotes = yf.Search(query, max_results=8).quotes
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
    except Exception:
        pass

    if not results:
        query_lower = query.lower()
        for name_fragment, (ticker, display_name) in _KNOWN_TICKERS.items():
            if name_fragment in query_lower or query_lower in name_fragment:
                results.append({"symbol": ticker, "name": display_name, "exchange": "", "type": "EQUITY"})

    return results


def _load_data(symbol: str, tf_conf: dict, demo_mode: bool, uploaded) -> tuple[pd.DataFrame | None, str, str | None]:
    """Returns (df, label, error) for one timeframe config."""
    if uploaded is not None:
        try:
            return _load_csv(uploaded), "CSV importé", None
        except Exception as exc:
            return None, "", str(exc)
    if demo_mode:
        freq = {"5min": "5min", "15min": "15min", "1H": "1h", "1D": "1D"}[tf_conf["bar_size"]]
        n_bars = 600 if tf_conf["bar_size"] in ("5min", "15min") else 400
        df = _generate_synthetic_ohlcv(n_bars, freq, "mean_reverting", seed=hash(symbol) % 1000)
        return df, f"{symbol.upper()} (démo synthétique)", None
    try:
        df = _fetch_yfinance(symbol.strip().upper(), tf_conf["start"], tf_conf["bar_size"])
        return df, f"{symbol.upper()} (yfinance)", None
    except Exception as exc:
        return None, "", str(exc)


# ---------------------------------------------------------------------------
# Signal + chart helpers (shared between the main detailed view and the
# multi-timeframe strip, so both stay in sync with the same logic).
# ---------------------------------------------------------------------------

def _run_three_tier(df: pd.DataFrame, tf_conf: dict, lookback: int, entry_z: float, exit_z: float, fast: int, slow: int) -> dict:
    mr = MeanReversionSignal(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
    mom = MomentumSignal(fast=fast, slow=slow)
    pa = PriceActionConfluenceSignal(htf_rule=tf_conf["htf_rule"])

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
    return dict(
        mr_result=mr_result, mom_result=mom_result, pa_result=pa_result,
        last_mr=last_mr, last_mom=last_mom, last_pa=last_pa,
        active_tier=active_tier, primary_side=primary_side, confidence=confidence,
    )


def _compute_levels(df: pd.DataFrame, primary_side: int, atr_stop_mult: float, atr_tp_mult: float, capital: float):
    price = float(df["close"].iloc[-1])
    atr = float(average_true_range(df).iloc[-1])
    stop = tp = None
    if primary_side != 0:
        limits = RiskLimits(atr_stop_multiple=atr_stop_mult, atr_tp_multiple=atr_tp_mult)
        rm = RiskManager(limits=limits, starting_equity=capital)
        stop = rm.compute_stop_price(price, atr, primary_side)
        tp = rm.compute_take_profit_price(price, atr, primary_side)
    return price, atr, stop, tp


def _compute_rangebreaks(window: pd.DataFrame) -> list:
    """Nights/weekends/closures show up as flat dead space on a
    continuous time axis, squeezing real candles into sparse-looking
    clusters. Derive the actual gaps from this window's own bar
    spacing (works for any bar size or exchange hours) and skip them."""
    inferred_freq = window.index.to_series().diff().median()
    if pd.notna(inferred_freq) and inferred_freq > pd.Timedelta(0):
        full_range = pd.date_range(window.index.min(), window.index.max(), freq=inferred_freq)
        observed = set(window.index)
        missing = [ts for ts in full_range if ts not in observed]
        if missing:
            return [dict(values=missing)]
    return []


def _build_chart(
    window: pd.DataFrame, tiers: dict, primary_side: int, price: float, stop: float | None, tp: float | None,
    badge_color: str, show_rsi: bool = True, height: int = 640,
) -> go.Figure:
    rangebreaks = _compute_rangebreaks(window)
    rows = 2 if show_rsi else 1
    row_heights = [0.75, 0.25] if show_rsi else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=show_rsi, row_heights=row_heights, vertical_spacing=0.05)
    fig.add_trace(
        go.Candlestick(
            x=window.index, open=window["open"], high=window["high"], low=window["low"], close=window["close"],
            name="Prix", showlegend=False, increasing_line_color=GOOD, decreasing_line_color=CRITICAL,
            increasing_fillcolor=GOOD, decreasing_fillcolor=CRITICAL,
        ),
        row=1, col=1,
    )

    # Only the CURRENT signal gets a marker -- one big, unambiguous arrow on
    # the last bar, not a triangle on every historical flip of the three
    # tiers. Marking every past crossover is what made the chart read as
    # cluttered noise instead of "here is today's call," which a real
    # broker app never shows either.
    marker_size = 30 if show_rsi else 18
    last_ts = window.index[-1:]
    if primary_side == 1:
        fig.add_trace(
            go.Scatter(
                x=last_ts, y=[window["low"].iloc[-1] * 0.98], mode="markers", name="Achat",
                marker=dict(color=GOOD, symbol="triangle-up", size=marker_size, line=dict(width=2, color="white")),
            ),
            row=1, col=1,
        )
    elif primary_side == -1:
        fig.add_trace(
            go.Scatter(
                x=last_ts, y=[window["high"].iloc[-1] * 1.02], mode="markers", name="Vente",
                marker=dict(color=CRITICAL, symbol="triangle-down", size=marker_size, line=dict(width=2, color="white")),
            ),
            row=1, col=1,
        )

    # Labels carry only the word, not the number -- the exact entry/stop/TP
    # prices already sit in the metric cards above the chart, so repeating
    # them here just risked clipping against the right edge on narrower
    # screens (and doubled up information the reader already has).
    if primary_side != 0 and stop is not None and tp is not None:
        fig.add_hline(
            y=price, line_dash="solid", line_color=badge_color, opacity=0.9, line_width=2,
            annotation_text="Entrée", annotation_position="right",
            annotation_font=dict(color=badge_color, size=11), row=1, col=1,
        )
        fig.add_hline(
            y=stop, line_dash="dash", line_color=CRITICAL, opacity=0.7, line_width=1.5,
            annotation_text="Stop", annotation_position="right",
            annotation_font=dict(color=CRITICAL, size=11), row=1, col=1,
        )
        fig.add_hline(
            y=tp, line_dash="dash", line_color=GOOD, opacity=0.7, line_width=1.5,
            annotation_text="Objectif", annotation_position="right",
            annotation_font=dict(color=GOOD, size=11), row=1, col=1,
        )

    if show_rsi:
        rsi_series = compute_rsi(window["close"])
        fig.add_trace(
            go.Scatter(x=window.index, y=rsi_series, name="RSI", showlegend=False, line=dict(color=TIER_COLORS["Price-action"], width=1.5)),
            row=2, col=1,
        )

    fig.update_layout(
        height=height, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=10, b=10, l=10, r=64), dragmode="zoom",
    )
    if show_rsi:
        # Rangeslider goes on the bottom-most (RSI) row only -- putting
        # it on row 1 as well (or relying on go.Candlestick's own
        # default, which is enabled) makes it render *between* the two
        # rows instead of below both.
        fig.update_xaxes(rangeslider=dict(visible=False), rangebreaks=rangebreaks, row=1, col=1)
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08), rangebreaks=rangebreaks, row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    else:
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.12), rangebreaks=rangebreaks, row=1, col=1)
    fig.update_yaxes(title_text="Prix", fixedrange=False, row=1, col=1)
    return fig


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
    df, label, load_error = _load_data(symbol, tf_conf, demo_mode, uploaded)

    if load_error:
        st.error(
            f"Impossible de charger **{symbol.upper()}** en direct : {load_error}\n\n"
            "Vérifie l'orthographe du symbole, réessaie dans une minute (Yahoo Finance "
            "bloque parfois temporairement les requêtes), ou coche **Mode démo "
            "(hors-ligne)** pour tester le scanner sans dépendre du réseau."
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

    with st.spinner("Analyse en cours (le test ADF de stationnarité peut prendre quelques secondes)..."):
        tiers = _run_three_tier(df, tf_conf, lookback, entry_z, exit_z, fast, slow)

    primary_side = tiers["primary_side"]
    active_tier = tiers["active_tier"]
    confidence = tiers["confidence"]
    price, atr, stop, tp = _compute_levels(df, primary_side, atr_stop_mult, atr_tp_mult, capital)
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
            (t1, "Mean-reversion", tiers["last_mr"]),
            (t2, "Momentum", tiers["last_mom"]),
            (t3, "Price-action", tiers["last_pa"]),
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

    # -- Candlestick chart with RSI panel --

    st.subheader("Bougies et RSI")
    st.caption(
        "▲ vert / ▼ rouge = le signal actuel (achat ou vente), affiché une seule fois sur "
        "la dernière bougie -- pas un historique de tous les signaux passés. Ligne pleine = "
        "entrée, pointillé rouge = stop, pointillé vert = objectif (take-profit). Boutons "
        "+/- ou glisser-sélectionner pour zoomer le temps, curseur sous le graphique pour "
        "naviguer, glisser directement sur les prix à droite pour zoomer l'échelle -- la "
        "molette ne fait pas défiler le graphique, elle fait défiler la page."
    )

    window = df.tail(min(200, len(df)))
    fig = _build_chart(window, tiers, primary_side, price, stop, tp, badge_color, show_rsi=True, height=640)
    st.plotly_chart(
        fig, use_container_width=True, key="main_chart",
        config={"scrollZoom": False, "displaylogo": False},
    )

    # -- Multi-timeframe strip: the same decision, at a glance, across
    # every preset timeframe (not just the one selected above) --

    st.subheader("Vue multi-horizons")
    st.caption("Le même symbole, analysé sur chacun des 4 horizons -- pratique pour repérer un signal qui n'apparaît que sur certains timeframes.")

    mini_cols = st.columns(2)
    for i, (mini_label, mini_conf) in enumerate(TIMEFRAME_OPTIONS.items()):
        with mini_cols[i % 2]:
            if mini_label == tf_label:
                mini_df, mini_tiers, mini_primary = df, tiers, primary_side
                mini_price, mini_stop, mini_tp = price, stop, tp
                mini_error = None
            else:
                mini_df, _, mini_error = _load_data(symbol, mini_conf, demo_mode, uploaded)
                if not mini_error and mini_df is not None and not mini_df.empty:
                    mini_df = DataCleaner.clean(mini_df)
                    if len(mini_df) < max(slow, lookback) + 10:
                        mini_error = "historique insuffisant pour ces paramètres"
                if not mini_error:
                    mini_tiers = _run_three_tier(mini_df, mini_conf, lookback, entry_z, exit_z, fast, slow)
                    mini_primary = mini_tiers["primary_side"]
                    mini_price, _, mini_stop, mini_tp = _compute_levels(mini_df, mini_primary, atr_stop_mult, atr_tp_mult, capital)

            mini_badge_color = {1: GOOD, -1: CRITICAL, 0: MUTED}[mini_primary] if not mini_error else MUTED
            mini_badge_text = {1: "ACHAT", -1: "VENTE", 0: "NEUTRE"}.get(mini_primary, "N/D") if not mini_error else "ERREUR"

            st.markdown(
                f'<div class="bb-mini-title">{mini_label} '
                f'<span class="bb-badge-sm" style="background:{mini_badge_color};">{mini_badge_text}</span></div>',
                unsafe_allow_html=True,
            )
            if mini_error:
                st.caption(f"Indisponible : {mini_error}")
            else:
                mini_window = mini_df.tail(min(120, len(mini_df)))
                mini_fig = _build_chart(
                    mini_window, mini_tiers, mini_primary, mini_price, mini_stop, mini_tp,
                    mini_badge_color, show_rsi=False, height=300,
                )
                st.plotly_chart(
                    mini_fig, use_container_width=True, key=f"mini_chart_{i}",
                    config={"scrollZoom": False, "displaylogo": False},
                )

    # -- Advanced sections: meta-model, full backtest, raw data --

    if train_meta:
        with st.expander("🤖 Meta-modèle ML (Random Forest + PurgedKFold)", expanded=True):
            with st.spinner("Entraînement (validation croisée purgée)..."):
                primary = tiers["mr_result"].side[tiers["mr_result"].side != 0]
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
