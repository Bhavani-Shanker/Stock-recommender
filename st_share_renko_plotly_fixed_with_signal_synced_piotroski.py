"""
Stock Technical Analysis & Recommendation App (Streamlit)

ADDITIONS (WITHOUT BREAKING EXISTING FEATURES / LOGIC):
------------------------------------------------------
1) Price line is now clearly visible on every chart:
   - Price (Close) is always plotted with EMA/VWAP/MVWAP/Bollinger in LINE charts
   - Volume is shown separately as a BAR chart (so it doesn't flatten price)

2) Each chart shows its own HOLD/BUY/SELL label to make interpretation clear.

SETUP:
------
pip install streamlit yfinance pandas ta numpy
streamlit run stock_recommender_app.py
"""

import datetime
from typing import Dict, Any, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

import plotly.graph_objects as go


# ============================================================
# SYMBOL LISTS (US + NSE) — ADDITIVE (does not change analysis logic)
# ============================================================
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_us_symbols() -> pd.DataFrame:
    """Load a broad list of US tickers (NASDAQ + other listed) with names.

    Data source: NasdaqTrader symbol directory files.
    - https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
    Returns columns: Symbol, Name, Exchange
    """
    urls = [
        # NASDAQ-listed symbols
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        # "Other Listed" includes NYSE/AMEX and other venues
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    frames = []
    for url in urls:
        try:
            df = pd.read_csv(url, sep="|", dtype=str)
            # Drop footer rows (e.g., File Creation Time / Number of Records)
            if "Symbol" in df.columns:
                df = df[df["Symbol"].notna() & (df["Symbol"].str.upper() != "FILE CREATION TIME")]
            elif "ACT Symbol" in df.columns:
                df = df[df["ACT Symbol"].notna() & (df["ACT Symbol"].str.upper() != "FILE CREATION TIME")]

            if "Symbol" in df.columns:
                symbol_col = "Symbol"
                name_col = "Security Name" if "Security Name" in df.columns else "Security Name"
                exch_col = "Exchange" if "Exchange" in df.columns else None
            else:
                symbol_col = "ACT Symbol"
                name_col = "Security Name" if "Security Name" in df.columns else "Security Name"
                exch_col = "Exchange" if "Exchange" in df.columns else None

            out = pd.DataFrame({
                "Symbol": df[symbol_col].astype(str).str.strip(),
                "Name": df[name_col].astype(str).str.strip(),
                # For NASDAQ-listed file, exchange is implicitly NASDAQ.
                # For otherlisted file, map NasdaqTrader exchange codes (N=NYSE, A=AMEX, P=ARCA, etc.)
                "Exchange": (
                    "NASDAQ"
                    if (exch_col is None)
                    else df[exch_col].astype(str).str.strip().replace(
                        {
                            "N": "NYSE",
                            "A": "AMEX",
                            "P": "ARCA",
                            "Z": "BATS",
                            "V": "IEX",
                        }
                    )
                ),
            })
            # Remove obviously invalid / test symbols
            out = out[out["Symbol"].str.match(r"^[A-Z0-9\.\-\^=]+$")]
            frames.append(out)
        except Exception:
            continue

    if not frames:
        # Minimal fallback list (keeps app usable even if symbol download fails)
        return pd.DataFrame(
            {"Symbol": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"],
             "Name": ["Apple Inc.", "Microsoft Corp.", "NVIDIA Corp.", "Alphabet Inc. (Class A)", "Amazon.com Inc.", "Tesla Inc."],
             "Exchange": ["NASDAQ"] * 6}
        )

    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Symbol"])
    all_df["Name"] = all_df["Name"].fillna("")
    all_df = all_df.sort_values(["Name", "Symbol"], kind="mergesort").reset_index(drop=True)
    return all_df


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_nse_symbols() -> pd.DataFrame:
    """Load NSE equity symbols with company names.

    Data source: NSE "EQUITY_L.csv" (public archive).
    Returns columns: Symbol, Name
    """
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = pd.read_csv(url, dtype=str)
        # Common columns: SYMBOL, NAME OF COMPANY
        sym = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
        name = "NAME OF COMPANY" if "NAME OF COMPANY" in df.columns else df.columns[1]
        out = pd.DataFrame({
            "Symbol": df[sym].astype(str).str.strip(),
            "Name": df[name].astype(str).str.strip(),
        })
        out = out[out["Symbol"].str.match(r"^[A-Z0-9&\-\.]+$")]
        out = out.sort_values(["Name", "Symbol"], kind="mergesort").reset_index(drop=True)
        return out
    except Exception:
        # Minimal fallback list
        return pd.DataFrame(
            {"Symbol": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"],
             "Name": ["Reliance Industries", "Tata Consultancy Services", "Infosys", "HDFC Bank", "ICICI Bank", "State Bank of India"]}
        )


def _ticker_picker_ui():
    """Additive UI: 2 tabs for US & NSE selection.
    Returns: selected_ticker (str) or '' if not selected.
    """
    tab_us, tab_in = st.tabs(["🇺🇸 American Stocks", "🇮🇳 Indian Stocks (NSE)"])

    selected = ""

    with tab_us:
        st.caption("Pick from a searchable US stock list (NASDAQ/NYSE/AMEX).")
        us_df = load_us_symbols()
        q = st.text_input("Search US stocks (name or ticker)", value="", key="us_search").strip().upper()
        view = us_df
        if q:
            mask = view["Symbol"].str.contains(q, na=False) | view["Name"].str.upper().str.contains(q, na=False)
            view = view[mask]
        # Keep the widget snappy: cap list size shown
        view = view.head(5000)

        if not view.empty:
            options = (view["Name"].fillna("") + " — " + view["Symbol"]).tolist()
            pick = st.selectbox("Select a US stock", options, index=0, key="us_pick")
            selected = pick.split(" — ")[-1].strip()
        else:
            st.info("No matches. Try a different search term.")

    with tab_in:
        st.caption('Pick from NSE stock list. The app will auto-suffix the ticker with ".NS".')
        in_df = load_nse_symbols()
        q = st.text_input("Search NSE stocks (name or symbol)", value="", key="in_search").strip().upper()
        view = in_df
        if q:
            mask = view["Symbol"].str.contains(q, na=False) | view["Name"].str.upper().str.contains(q, na=False)
            view = view[mask]
        view = view.head(5000)

        if not view.empty:
            options = (view["Name"].fillna("") + " — " + view["Symbol"] + ".NS").tolist()
            pick = st.selectbox("Select an NSE stock", options, index=0, key="in_pick")
            selected = pick.split(" — ")[-1].strip()  # already has .NS
        else:
            st.info("No matches. Try a different search term.")

    return selected


# ============================================================
# MARKET PICKS (Best Buy / Hold / Sell) — ADDITIVE
# ============================================================
def _analyze_single_symbol_for_snapshot(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Small wrapper used by Market Picks snapshot.

    IMPORTANT: Reuses your *existing* pipeline:
      fetch_stock_data -> add_technical_indicators -> compute_indicator_signals -> derive_final_recommendation

    Returns a dict safe for DataFrame rows.
    """
    try:
        df = fetch_stock_data(symbol, period=period, interval="1d")
        if df.empty:
            return {"Ticker": symbol, "Score": np.nan, "Recommendation": "N/A", "Reason": "No data"}
        df = add_technical_indicators(df)
        _, score = compute_indicator_signals(df)
        rec, reason = derive_final_recommendation(score)
        last_close = df["Close"].iloc[-1] if "Close" in df.columns and len(df) else np.nan

        # ADDITIVE: Piotroski F-Score (fundamentals) — does not change technical scoring
        p_score, _p_details = compute_piotroski_f_score(symbol)

        return {
            "Ticker": symbol,
            "Last Close": float(last_close) if pd.notna(last_close) else np.nan,
            "Score": int(score),
            "Piotroski F-Score": (int(p_score) if p_score is not None else np.nan),
            "Recommendation": rec,
            "Reason": reason,
        }
    except Exception as e:
        return {"Ticker": symbol, "Score": np.nan, "Recommendation": "N/A", "Reason": str(e)[:120]}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def build_market_snapshot(symbols: Tuple[str, ...], period: str = "1y", max_workers: int = 6) -> pd.DataFrame:
    """Compute BUY/HOLD/SELL snapshot for a list of symbols.

    - Cached for 1 hour to keep the app responsive.
    - Uses a thread pool to speed up IO-bound yfinance calls.
    """
    max_workers = max(1, int(max_workers))
    rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_analyze_single_symbol_for_snapshot, sym, period): sym for sym in symbols}
        for fut in as_completed(futs):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    if not df.empty and "Score" in df.columns:
        df = df.sort_values(["Recommendation", "Score"], ascending=[True, False], kind="mergesort")
    return df




# ============================================================
# TOP-N PREFILTER (Market Cap / Volume) — ADDITIVE
# ============================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _fetch_symbol_metric(symbol: str, metric: str) -> float:
    """Fetch a single metric for a symbol using yfinance.

    metric:
      - 'market_cap' : market capitalization
      - 'avg_volume' : average daily volume (approx)

    Uses fast_info when available (faster), falls back to info/history.
    Returns np.nan when unavailable.
    """
    try:
        t = yf.Ticker(symbol)
        # Prefer fast_info (much faster than info)
        fi = getattr(t, "fast_info", None)
        if fi:
            if metric == "market_cap":
                v = fi.get("market_cap") or fi.get("marketCap")
                if v is not None:
                    return float(v)
            if metric == "avg_volume":
                # Prefer 'tenDayAverageVolume' if present
                v = fi.get("ten_day_average_volume") or fi.get("tenDayAverageVolume") or fi.get("averageVolume")
                if v is not None:
                    return float(v)

        # Fallback to info (can be slower, sometimes blocked)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        if metric == "market_cap":
            v = info.get("marketCap")
            if v is not None:
                return float(v)

        if metric == "avg_volume":
            v = info.get("averageVolume") or info.get("averageDailyVolume10Day")
            if v is not None:
                return float(v)

        # Final fallback: compute average volume from recent history
        if metric == "avg_volume":
            try:
                h = t.history(period="3mo", interval="1d", auto_adjust=False)
                if not h.empty and "Volume" in h.columns:
                    return float(pd.to_numeric(h["Volume"], errors="coerce").dropna().tail(30).mean())
            except Exception:
                pass

    except Exception:
        pass

    return float("nan")



# ============================================================
# PIOTROSKI F-SCORE (ADDITIVE)
# - Best-effort fundamental score (0..9) using yfinance annual statements.
# - Does NOT change any existing technical scoring or recommendations.
# ============================================================
@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def compute_piotroski_f_score(symbol: str) -> Tuple[Optional[int], Dict[str, Any]]:
    """Compute Piotroski F-Score (0..9) using annual financial statements.

    Notes / limitations:
    - Relies on yfinance availability for income statement, balance sheet, cashflow.
    - Some symbols (especially smaller caps / non-US listings) may have missing fields.
    - If critical data is missing, returns (None, details) instead of a potentially misleading score.
    """
    try:
        t = yf.Ticker(symbol)

        # yfinance returns DataFrames with columns as period end dates (most recent first)
        inc = getattr(t, "financials", None)  # Income Statement (annual)
        bal = getattr(t, "balance_sheet", None)  # Balance Sheet (annual)
        cf = getattr(t, "cashflow", None)  # Cash Flow (annual)

        if inc is None or bal is None or cf is None:
            return None, {"error": "No financial statements available"}

        # Normalize to DataFrame
        inc = inc.copy() if hasattr(inc, "copy") else pd.DataFrame()
        bal = bal.copy() if hasattr(bal, "copy") else pd.DataFrame()
        cf = cf.copy() if hasattr(cf, "copy") else pd.DataFrame()

        # Need at least 2 periods for most comparisons
        if inc.empty or bal.empty or cf.empty or inc.shape[1] < 2 or bal.shape[1] < 2 or cf.shape[1] < 2:
            return None, {"error": "Insufficient annual statement history"}

        # Helper: safe get by row name (case-insensitive match)
        def _row(df: pd.DataFrame, candidates: List[str]):
            if df is None or df.empty:
                return None
            idx_map = {str(i).strip().lower(): i for i in df.index}
            for c in candidates:
                key = str(c).strip().lower()
                if key in idx_map:
                    return pd.to_numeric(df.loc[idx_map[key]], errors="coerce")
            # try fuzzy contains match
            for i in df.index:
                name = str(i).strip().lower()
                if any(key in name for key in [str(c).strip().lower() for c in candidates]):
                    return pd.to_numeric(df.loc[i], errors="coerce")
            return None

        # Pull series (most recent = [0], prior = [1])
        net_income = _row(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
        total_assets = _row(bal, ["Total Assets", "TotalAssets"])
        op_cfo = _row(cf, ["Total Cash From Operating Activities", "Operating Cash Flow", "Net Cash Provided By Operating Activities"])

        long_term_debt = _row(bal, ["Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation"])
        current_assets = _row(bal, ["Total Current Assets", "Current Assets"])
        current_liab = _row(bal, ["Total Current Liabilities", "Current Liabilities"])

        revenue = _row(inc, ["Total Revenue", "TotalRevenue", "Revenue"])
        gross_profit = _row(inc, ["Gross Profit", "GrossProfit"])

        # Shares outstanding (often missing in statements); try a few balance sheet rows
        shares = _row(bal, ["Ordinary Shares Number", "Common Stock Shares Outstanding", "Share Issued"])

        # Guardrails: need these for baseline calculations
        def _val(series, i):
            try:
                v = series.iloc[i]
                return None if pd.isna(v) else float(v)
            except Exception:
                return None

        ni0, ni1 = _val(net_income, 0), _val(net_income, 1)
        ta0, ta1 = _val(total_assets, 0), _val(total_assets, 1)
        cfo0, cfo1 = _val(op_cfo, 0), _val(op_cfo, 1)

        if None in (ni0, ni1, ta0, ta1, cfo0, cfo1) or ta0 == 0 or ta1 == 0:
            return None, {"error": "Missing key fields for ROA/CFO"}

        roa0 = ni0 / ta0
        roa1 = ni1 / ta1

        # Start scoring
        score = 0
        details: Dict[str, Any] = {}

        # 1) Positive ROA
        details["ROA_Positive"] = roa0 > 0
        score += 1 if roa0 > 0 else 0

        # 2) Positive CFO
        details["CFO_Positive"] = cfo0 > 0
        score += 1 if cfo0 > 0 else 0

        # 3) ROA improvement
        details["ROA_Improved"] = roa0 > roa1
        score += 1 if roa0 > roa1 else 0

        # 4) Accruals (CFO > Net Income)
        details["Accruals_CFO_gt_NI"] = cfo0 > ni0
        score += 1 if cfo0 > ni0 else 0

        # 5) Lower leverage (LT Debt / Assets decreased)
        ltd0, ltd1 = _val(long_term_debt, 0), _val(long_term_debt, 1)
        if None not in (ltd0, ltd1) and ta0 and ta1:
            lev0 = ltd0 / ta0
            lev1 = ltd1 / ta1
            details["Leverage_Down"] = lev0 < lev1
            score += 1 if lev0 < lev1 else 0
        else:
            details["Leverage_Down"] = None

        # 6) Higher current ratio
        ca0, ca1 = _val(current_assets, 0), _val(current_assets, 1)
        cl0, cl1 = _val(current_liab, 0), _val(current_liab, 1)
        if None not in (ca0, ca1, cl0, cl1) and cl0 != 0 and cl1 != 0:
            cr0 = ca0 / cl0
            cr1 = ca1 / cl1
            details["CurrentRatio_Up"] = cr0 > cr1
            score += 1 if cr0 > cr1 else 0
        else:
            details["CurrentRatio_Up"] = None

        # 7) No new shares issued (shares outstanding not increased)
        sh0, sh1 = _val(shares, 0), _val(shares, 1)
        if None not in (sh0, sh1) and sh0 > 0 and sh1 > 0:
            details["No_Dilution"] = sh0 <= sh1
            score += 1 if sh0 <= sh1 else 0
        else:
            details["No_Dilution"] = None

        # 8) Higher gross margin
        rev0, rev1 = _val(revenue, 0), _val(revenue, 1)
        gp0, gp1 = _val(gross_profit, 0), _val(gross_profit, 1)
        if None not in (rev0, rev1, gp0, gp1) and rev0 != 0 and rev1 != 0:
            gm0 = gp0 / rev0
            gm1 = gp1 / rev1
            details["GrossMargin_Up"] = gm0 > gm1
            score += 1 if gm0 > gm1 else 0
        else:
            details["GrossMargin_Up"] = None

        # 9) Higher asset turnover (Revenue / Assets)
        if None not in (rev0, rev1, ta0, ta1) and ta0 != 0 and ta1 != 0:
            at0 = rev0 / ta0
            at1 = rev1 / ta1
            details["AssetTurnover_Up"] = at0 > at1
            score += 1 if at0 > at1 else 0
        else:
            details["AssetTurnover_Up"] = None

        return int(score), details

    except Exception as e:
        return None, {"error": str(e)[:200]}


def top_n_symbols_by_metric(symbols: List[str], metric: str, top_n: int, cap: int = 1500) -> List[str]:
    """Return top N symbols by metric (descending).

    - 'cap' limits how many symbols we attempt (protects app responsiveness).
    - Uses caching, but first-time calls for large universes can still take time.
    """
    if not symbols:
        return []
    metric = metric.lower().strip()
    if metric not in {"market_cap", "avg_volume"}:
        return symbols[:top_n]

    universe = symbols[: min(len(symbols), int(cap))]
    rows = []
    for sym in universe:
        rows.append((sym, _fetch_symbol_metric(sym, metric)))
    dfm = pd.DataFrame(rows, columns=["Symbol", "Metric"])
    dfm["Metric"] = pd.to_numeric(dfm["Metric"], errors="coerce")
    dfm = dfm.dropna(subset=["Metric"]).sort_values("Metric", ascending=False, kind="mergesort")
    if dfm.empty:
        return symbols[:top_n]
    return dfm["Symbol"].head(int(top_n)).tolist()


def render_market_picks_ui():
    """UI block: Market Picks section (US + NSE tabs)."""
    st.markdown("---")
    st.header("📌 Market Picks: Best Buy / Hold / Sell")

    market_us, market_in = st.tabs(["🇺🇸 US Market Picks", "🇮🇳 NSE Market Picks"])

    # Common controls
    def _controls(scope_key: str):
        c1, c2, c3 = st.columns([1, 1, 1])
        n_scan = c1.slider("How many stocks to scan", min_value=10, max_value=300, value=60, step=10, key=f"nscan_{scope_key}")
        max_workers = c2.slider("Threads", min_value=1, max_value=12, value=6, step=1, key=f"workers_{scope_key}")
        period = c3.selectbox("Snapshot period", ["6mo", "1y", "2y"], index=1, key=f"period_{scope_key}")

        # ADDITIVE: optional Top-N prefilter
        p1, p2 = st.columns([1, 1])
        prefilter = p1.selectbox(
            "Optional: Pick only Top-N by…",
            ["None", "Market Cap", "Volume"],
            index=0,
            key=f"prefilter_{scope_key}",
        )
        top_n = p2.slider("Top N", min_value=10, max_value=500, value=int(n_scan), step=10, key=f"topn_{scope_key}")

        return n_scan, max_workers, period, prefilter, top_n

    with market_us:
        st.caption("Scans a subset of US tickers and ranks them by your existing signal score.")
        q = st.text_input("Filter (optional) — US name or ticker", value="", key="us_market_filter").strip().upper()
        n_scan, max_workers, period, prefilter, top_n = _controls("us")

        symbols_df = load_us_symbols()
        if q:
            mask = symbols_df["Symbol"].str.contains(q, na=False) | symbols_df["Name"].str.upper().str.contains(q, na=False)
            symbols_df = symbols_df[mask]

        # ADDITIVE: Exchange filter (All / NASDAQ / NYSE)
        exchange_choice = st.selectbox(
            "US Exchange",
            ["All", "NASDAQ", "NYSE"],
            index=0,
            key="us_exchange_choice",
            help="Filters the US universe list only (does not change any indicator logic).",
        )
        if exchange_choice != "All" and "Exchange" in symbols_df.columns:
            symbols_df = symbols_df[symbols_df["Exchange"].astype(str).str.upper().str.contains(exchange_choice)]

        # ADDITIVE: Market-cap-weighted ranking toggle (keeps original Score intact)
        use_mcap_weight = st.toggle(
            "Use market-cap-weighted ranking",
            value=False,
            key="us_use_mcap_weight",
            help="When ON, rankings use a WeightedScore = Score × (1 + α·normalized_market_cap). Score stays unchanged.",
        )
        alpha = st.slider(
            "Market-cap weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
            key="us_mcap_alpha",
            disabled=not use_mcap_weight,
        )

        symbols_list = symbols_df["Symbol"].tolist()

        if prefilter != "None":
            metric = "market_cap" if prefilter == "Market Cap" else "avg_volume"
            symbols_list = top_n_symbols_by_metric(symbols_list, metric=metric, top_n=int(top_n))
        else:
            symbols_list = symbols_list[: int(n_scan)]

        symbols = tuple(symbols_list)

        if st.button("Generate US Picks", key="btn_us_picks"):
            with st.spinner("Building US snapshot…"):
                snap = build_market_snapshot(symbols, period=period, max_workers=max_workers)

            if snap.empty:
                st.warning("No data returned.")
            else:
                # Optional: market-cap-weighted ranking (additive)
                sort_col = "Score"
                if use_mcap_weight:
                    # Fetch market caps for symbols in the snapshot and compute a WeightedScore
                    snap = snap.copy()
                    caps = []
                    for sym in snap["Ticker"].astype(str).tolist():
                        caps.append(_fetch_symbol_metric(sym, "market_cap"))
                    snap["Market Cap"] = pd.to_numeric(caps, errors="coerce")
                    # Normalize market cap to 0..1 (robust to NaNs)
                    cap_min = snap["Market Cap"].min(skipna=True)
                    cap_max = snap["Market Cap"].max(skipna=True)
                    if pd.notna(cap_min) and pd.notna(cap_max) and cap_max != cap_min:
                        cap_norm = (snap["Market Cap"] - cap_min) / (cap_max - cap_min)
                    else:
                        cap_norm = 0.0
                    snap["WeightedScore"] = pd.to_numeric(snap["Score"], errors="coerce") * (1.0 + float(alpha) * cap_norm)
                    sort_col = "WeightedScore"

                buy_df = snap[snap["Recommendation"] == "BUY"].sort_values(sort_col, ascending=False).head(25)
                hold_df = snap[snap["Recommendation"] == "HOLD"].sort_values(sort_col, ascending=False).head(25)
                sell_df = snap[snap["Recommendation"] == "SELL"].sort_values(sort_col, ascending=True).head(25)

                t1, t2, t3 = st.tabs(["✅ Best shares to BUY", "🟡 Best shares to HOLD", "❌ Worst shares to SELL"])
                with t1:
                    st.dataframe(buy_df.reset_index(drop=True), use_container_width=True)
                with t2:
                    st.dataframe(hold_df.reset_index(drop=True), use_container_width=True)
                with t3:
                    st.dataframe(sell_df.reset_index(drop=True), use_container_width=True)

    with market_in:
        st.caption("Scans a subset of NSE tickers (auto-suffixed with .NS) and ranks them by your existing signal score.")
        q = st.text_input("Filter (optional) — NSE name or symbol", value="", key="in_market_filter").strip().upper()
        n_scan, max_workers, period, prefilter, top_n = _controls("in")

        symbols_df = load_nse_symbols()
        if q:
            mask = symbols_df["Symbol"].str.contains(q, na=False) | symbols_df["Name"].str.upper().str.contains(q, na=False)
            symbols_df = symbols_df[mask]

        symbols_list = (symbols_df["Symbol"].tolist())

        if prefilter != "None":
            metric = "market_cap" if prefilter == "Market Cap" else "avg_volume"
            # Apply .NS after picking top-N
            picked = top_n_symbols_by_metric(symbols_list, metric=metric, top_n=int(top_n))
            symbols_list = [s + ".NS" for s in picked]
        else:
            symbols_list = [(s + ".NS") for s in symbols_list[: int(n_scan)]]

        symbols = tuple(symbols_list)

        if st.button("Generate NSE Picks", key="btn_in_picks"):
            with st.spinner("Building NSE snapshot…"):
                snap = build_market_snapshot(symbols, period=period, max_workers=max_workers)

            if snap.empty:
                st.warning("No data returned.")
            else:
                buy_df = snap[snap["Recommendation"] == "BUY"].sort_values("Score", ascending=False).head(25)
                hold_df = snap[snap["Recommendation"] == "HOLD"].sort_values("Score", ascending=False).head(25)
                sell_df = snap[snap["Recommendation"] == "SELL"].sort_values("Score", ascending=True).head(25)

                t1, t2, t3 = st.tabs(["✅ Best shares to BUY", "🟡 Best shares to HOLD", "❌ Worst shares to SELL"])
                with t1:
                    st.dataframe(buy_df.reset_index(drop=True), use_container_width=True)
                with t2:
                    st.dataframe(hold_df.reset_index(drop=True), use_container_width=True)
                with t3:
                    st.dataframe(sell_df.reset_index(drop=True), use_container_width=True)




# ============================================================
# FETCH STOCK DATA
# ============================================================
def fetch_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================
# ADD TECHNICAL INDICATORS
# ============================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Existing indicators (unchanged)
    df["SMA_50"] = SMAIndicator(close, 50).sma_indicator()
    df["SMA_200"] = SMAIndicator(close, 200).sma_indicator()

    df["EMA_12"] = EMAIndicator(close, 12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close, 26).ema_indicator()

    df["RSI_14"] = RSIIndicator(close, 14).rsi()

    macd_ind = MACD(close)
    df["MACD"] = macd_ind.macd()
    df["MACD_signal"] = macd_ind.macd_signal()
    df["MACD_hist"] = macd_ind.macd_diff()

    bb = BollingerBands(close)
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    stoch = StochasticOscillator(high, low, close)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()

    atr = AverageTrueRange(high, low, close)
    df["ATR_14"] = atr.average_true_range()

    # Added indicators (already working in your version)
    df["EMA_20"] = EMAIndicator(close, 20).ema_indicator()
    df["EMA_34"] = EMAIndicator(close, 34).ema_indicator()
    df["EMA_50"] = EMAIndicator(close, 50).ema_indicator()

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3

    # VWAP
    df["VWAP"] = (df["Volume"] * typical_price).cumsum() / df["Volume"].cumsum()

    # MVWAP (20)
    df["MVWAP_20"] = (
        (typical_price * df["Volume"]).rolling(20).sum()
        / df["Volume"].rolling(20).sum()
    )

    return df


# ============================================================
# SAFE ACCESS
# ============================================================
def _safe(series: pd.Series, idx: int):
    try:
        v = series.iloc[idx]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


# ============================================================
# EXISTING SIGNAL LOGIC (UNCHANGED)
# ============================================================
def compute_indicator_signals(df: pd.DataFrame):
    if len(df) < 2:
        return {}, 0

    signals: Dict[str, Dict[str, Any]] = {}
    total = 0

    close_now = _safe(df["Close"], -1)

    # SMA trend
    sma50_now = _safe(df["SMA_50"], -1)
    sma200_now = _safe(df["SMA_200"], -1)
    sma50_prev = _safe(df["SMA_50"], -2)
    sma200_prev = _safe(df["SMA_200"], -2)

    t_score = 0
    if None not in (sma50_now, sma200_now, sma50_prev, sma200_prev, close_now):
        golden = sma50_prev <= sma200_prev and sma50_now > sma200_now
        death = sma50_prev >= sma200_prev and sma50_now < sma200_now
        up = sma50_now > sma200_now and close_now > sma50_now
        down = sma50_now < sma200_now and close_now < sma50_now

        if golden or up:
            t_score = 2
        elif death or down:
            t_score = -2
    signals["Trend (SMA 50/200)"] = {"score": t_score}
    total += t_score

    # EMA 12/26
    ema12 = _safe(df["EMA_12"], -1)
    ema26 = _safe(df["EMA_26"], -1)
    if ema12 is not None and ema26 is not None:
        e_score = 1 if ema12 > ema26 else -1 if ema12 < ema26 else 0
    else:
        e_score = 0
    signals["EMA 12/26"] = {"score": e_score}
    total += e_score

    # RSI
    rsi = _safe(df["RSI_14"], -1)
    if rsi is not None:
        r_score = 1 if rsi < 30 else -1 if rsi > 70 else 0
    else:
        r_score = 0
    signals["RSI"] = {"score": r_score}
    total += r_score

    # MACD
    macd = _safe(df["MACD"], -1)
    sig = _safe(df["MACD_signal"], -1)
    h_now = _safe(df["MACD_hist"], -1)
    h_prev = _safe(df["MACD_hist"], -2)

    if None not in (macd, sig, h_now, h_prev):
        m_score = 1 if (macd > sig and h_now > h_prev) else -1 if (macd < sig and h_now < h_prev) else 0
    else:
        m_score = 0
    signals["MACD"] = {"score": m_score}
    total += m_score

    # Bollinger
    bb_up = _safe(df["BB_upper"], -1)
    bb_low = _safe(df["BB_lower"], -1)
    if None not in (close_now, bb_up, bb_low):
        b_score = 1 if close_now < bb_low else -1 if close_now > bb_up else 0
    else:
        b_score = 0
    signals["Bollinger"] = {"score": b_score}
    total += b_score

    # Stochastic
    k_now = _safe(df["STOCH_K"], -1)
    d_now = _safe(df["STOCH_D"], -1)
    k_prev = _safe(df["STOCH_K"], -2)
    d_prev = _safe(df["STOCH_D"], -2)

    if None not in (k_now, d_now, k_prev, d_prev):
        s_score = 1 if (k_now < 20 and k_prev <= d_prev and k_now > d_now) else \
                  -1 if (k_now > 80 and k_prev >= d_prev and k_now < d_now) else 0
    else:
        s_score = 0
    signals["Stochastic"] = {"score": s_score}
    total += s_score

    # ATR
    atr = _safe(df["ATR_14"], -1)
    if atr is not None and close_now is not None and close_now != 0:
        pct = atr / close_now * 100
        a_score = 1 if pct < 2 else -1 if pct > 6 else 0
    else:
        a_score = 0
    signals["ATR"] = {"score": a_score}
    total += a_score

    return signals, total


def derive_final_recommendation(score: int) -> Tuple[str, str]:
    if score >= 2:
        return "BUY", f"Score {score} → bullish bias."
    elif score <= -2:
        return "SELL", f"Score {score} → bearish bias."
    return "HOLD", f"Score {score} → mixed / wait for clearer confirmation."


# ============================================================
# PER-CHART RECOMMENDATIONS (ADDITIVE)
# ============================================================
def label_box(rec: str, reason: str):
    """Show colored BUY/SELL/HOLD box."""
    if rec == "BUY":
        st.success(f"✅ **{rec}** — {reason}")
    elif rec == "SELL":
        st.error(f"❌ **{rec}** — {reason}")
    else:
        st.warning(f"⚖️ **{rec}** — {reason}")


def chart_rec_ema(df: pd.DataFrame) -> Tuple[str, str]:
    c = _safe(df["Close"], -1)
    e20 = _safe(df["EMA_20"], -1)
    e34 = _safe(df["EMA_34"], -1)
    e50 = _safe(df["EMA_50"], -1)
    if None in (c, e20, e34, e50):
        return "HOLD", "Not enough EMA data."

    if e20 > e34 > e50 and c > e20:
        return "BUY", "EMA20 > EMA34 > EMA50 and price above EMA20 (bullish stack)."
    if e20 < e34 < e50 and c < e20:
        return "SELL", "EMA20 < EMA34 < EMA50 and price below EMA20 (bearish stack)."
    return "HOLD", "EMAs not clearly stacked or price not confirming."


def chart_rec_bollinger(df: pd.DataFrame) -> Tuple[str, str]:
    c = _safe(df["Close"], -1)
    up = _safe(df["BB_upper"], -1)
    low = _safe(df["BB_lower"], -1)
    if None in (c, up, low):
        return "HOLD", "Not enough Bollinger data."

    if c < low:
        return "BUY", "Price below lower band (potential oversold bounce)."
    if c > up:
        return "SELL", "Price above upper band (potential overbought pullback)."
    return "HOLD", "Price inside bands (neutral)."


def chart_rec_vwap(df: pd.DataFrame) -> Tuple[str, str]:
    c = _safe(df["Close"], -1)
    v = _safe(df["VWAP"], -1)
    if None in (c, v):
        return "HOLD", "Not enough VWAP data."
    if c > v:
        return "BUY", "Price above VWAP (bullish intraday/period bias)."
    if c < v:
        return "SELL", "Price below VWAP (bearish intraday/period bias)."
    return "HOLD", "Price equals VWAP."


def chart_rec_mvwap(df: pd.DataFrame) -> Tuple[str, str]:
    c = _safe(df["Close"], -1)
    mv = _safe(df["MVWAP_20"], -1)
    if None in (c, mv):
        return "HOLD", "Not enough MVWAP data."
    if c > mv:
        return "BUY", "Price above MVWAP(20) (bullish bias)."
    if c < mv:
        return "SELL", "Price below MVWAP(20) (bearish bias)."
    return "HOLD", "Price equals MVWAP."


def chart_rec_macd(df: pd.DataFrame) -> Tuple[str, str]:
    m = _safe(df["MACD"], -1)
    s = _safe(df["MACD_signal"], -1)
    h_now = _safe(df["MACD_hist"], -1)
    h_prev = _safe(df["MACD_hist"], -2)
    if None in (m, s, h_now, h_prev):
        return "HOLD", "Not enough MACD data."

    if m > s and h_now > h_prev:
        return "BUY", "MACD above signal and histogram rising."
    if m < s and h_now < h_prev:
        return "SELL", "MACD below signal and histogram falling."
    return "HOLD", "MACD mixed."


def chart_rec_rsi(df: pd.DataFrame) -> Tuple[str, str]:
    r = _safe(df["RSI_14"], -1)
    if r is None:
        return "HOLD", "Not enough RSI data."
    if r < 30:
        return "BUY", f"RSI {r:.1f} oversold (<30)."
    if r > 70:
        return "SELL", f"RSI {r:.1f} overbought (>70)."
    return "HOLD", f"RSI {r:.1f} neutral (30–70)."


def chart_rec_stoch(df: pd.DataFrame) -> Tuple[str, str]:
    k_now = _safe(df["STOCH_K"], -1)
    d_now = _safe(df["STOCH_D"], -1)
    k_prev = _safe(df["STOCH_K"], -2)
    d_prev = _safe(df["STOCH_D"], -2)
    if None in (k_now, d_now, k_prev, d_prev):
        return "HOLD", "Not enough Stochastic data."

    bullish_cross = k_prev <= d_prev and k_now > d_now
    bearish_cross = k_prev >= d_prev and k_now < d_now

    if k_now < 20 and bullish_cross:
        return "BUY", "Bullish crossover in oversold zone (<20)."
    if k_now > 80 and bearish_cross:
        return "SELL", "Bearish crossover in overbought zone (>80)."
    return "HOLD", "Stochastic neutral/mixed."


def chart_rec_atr(df: pd.DataFrame) -> Tuple[str, str]:
    atr = _safe(df["ATR_14"], -1)
    c = _safe(df["Close"], -1)
    if None in (atr, c) or c == 0:
        return "HOLD", "Not enough ATR data."
    pct = atr / c * 100
    if pct > 6:
        return "HOLD", f"High volatility (ATR ~{pct:.2f}% of price) → risk elevated."
    if pct < 2:
        return "HOLD", f"Low volatility (ATR ~{pct:.2f}% of price) → stable."
    return "HOLD", f"Moderate volatility (ATR ~{pct:.2f}%)."



def chart_rec_renko(renko_df: pd.DataFrame, lookback: int = 5) -> Tuple[str, str, int]:
    """Return a simple BUY/SELL/HOLD label for Renko bricks + a score.

    IMPORTANT:
    - This is intentionally lightweight and *does not* alter your global score/recommendation logic.
    - It is used only for displaying a Renko-specific label (like other charts) and for adding a Renko row
      in the Indicator Signal Breakdown table.

    Logic (lookback bricks):
      - BUY  (+1): strong up-trend in bricks
      - SELL (-1): strong down-trend in bricks
      - HOLD (0): otherwise
    """
    if renko_df is None or renko_df.empty or "Direction" not in renko_df.columns:
        return "HOLD", "Renko unavailable (no bricks formed).", 0

    lb = max(2, int(lookback))
    d = pd.to_numeric(renko_df["Direction"], errors="coerce").dropna().tail(lb)

    if len(d) < 2:
        return "HOLD", "Not enough Renko bricks to infer trend.", 0

    up = int((d == 1).sum())
    down = int((d == -1).sum())

    if up == len(d):
        return "BUY", f"Renko shows {len(d)} consecutive UP bricks (strong bullish trend).", 1
    if down == len(d):
        return "SELL", f"Renko shows {len(d)} consecutive DOWN bricks (strong bearish trend).", -1

    momentum = int(d.sum())  # range [-lb, +lb]
    # Near-unanimous direction is still treated as a trend
    if momentum >= len(d) - 1:
        return "BUY", f"Renko momentum bullish (net +{momentum} over last {len(d)} bricks).", 1
    if momentum <= -(len(d) - 1):
        return "SELL", f"Renko momentum bearish (net {momentum} over last {len(d)} bricks).", -1

    return "HOLD", "Renko trend is mixed (no strong brick momentum).", 0


# ============================================================
# RENKO (ADDITIVE)
# - Default "brick size" uses ATR(5) (5-day volatility), but user can override.
# - Does NOT change any existing indicator logic or recommendations.
# ============================================================
def _compute_renko_brick_size(df: pd.DataFrame, atr_days: int, manual_brick: Optional[float] = None) -> float:
    """Return a Renko brick size.

    If manual_brick is provided and > 0, it is used directly.
    Otherwise, uses the latest ATR(atr_days) value as the brick size (common Renko practice).
    """
    if manual_brick is not None and float(manual_brick) > 0:
        return float(manual_brick)

    atr_days = max(1, int(atr_days))
    try:
        atr_ind = AverageTrueRange(df["High"], df["Low"], df["Close"], window=atr_days)
        atr_series = atr_ind.average_true_range()
        v = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else np.nan
        if pd.notna(v) and float(v) > 0:
            return float(v)
    except Exception:
        pass

    # Fallback: small fraction of last close if ATR unavailable (keeps UI usable)
    try:
        last_close = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
        return max(last_close * 0.01, 0.01)  # 1% fallback
    except Exception:
        return 1.0


def build_renko(df: pd.DataFrame, brick_size: float) -> pd.DataFrame:
    """Build Renko bricks from Close prices.

    Returns a DataFrame with:
      - BrickIndex: 1..N
      - BrickClose: brick close price
      - Direction: +1 (up) / -1 (down)
      - SourceTime: timestamp of the price that created the brick (best-effort)
    """
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return pd.DataFrame()

    brick = float(brick_size)
    if not np.isfinite(brick) or brick <= 0:
        return pd.DataFrame()

    bricks = []
    last = float(close.iloc[0])

    for t, price in close.items():
        price = float(price)

        # Add as many bricks as needed to "catch up" to price
        while price - last >= brick:
            last += brick
            bricks.append({"BrickClose": last, "Direction": 1, "SourceTime": t})

        while last - price >= brick:
            last -= brick
            bricks.append({"BrickClose": last, "Direction": -1, "SourceTime": t})

    if not bricks:
        # No bricks formed (very quiet period / huge brick size)
        return pd.DataFrame()

    out = pd.DataFrame(bricks)
    out.insert(0, "BrickIndex", np.arange(1, len(out) + 1))
    return out



def plot_renko_bricks(renko_df: pd.DataFrame, brick_size: float):
    """Render Renko bricks as true colored rectangles using Plotly shapes.

    - Green bricks: Direction = +1 (up)
    - Red bricks:   Direction = -1 (down)

    NOTE: This is purely a visualization helper and does not change any
    indicator logic, scoring, or recommendations.
    """
    if renko_df is None or renko_df.empty:
        return go.Figure()

    b = float(brick_size) if brick_size is not None else 0.0
    fig = go.Figure()

    # Add an invisible scatter so axes scale reliably
    fig.add_trace(
        go.Scatter(
            x=renko_df["BrickIndex"],
            y=renko_df["BrickClose"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for _, row in renko_df.iterrows():
        idx = float(row["BrickIndex"])
        close = float(row["BrickClose"])
        direction = int(row["Direction"])

        if direction == 1:
            y0, y1 = close - b, close
            color = "green"
        else:
            y0, y1 = close, close + b
            color = "red"

        fig.add_shape(
            type="rect",
            x0=idx - 0.48,
            x1=idx + 0.48,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line=dict(width=0),
            opacity=0.85,
            layer="above",
        )

    fig.update_layout(
        title="Renko Bricks",
        xaxis_title="Brick Index",
        yaxis_title="Price",
        showlegend=False,
        height=520,
        margin=dict(l=40, r=30, t=50, b=40),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    return fig


# ============================================================
# SIMPLE 30-DAY FORECAST (unchanged from your working version)
# ============================================================
def generate_30d_forecast(df: pd.DataFrame, horizon: int = 30) -> pd.Series:
    if df.empty or len(df) < 20:
        return pd.Series(dtype=float)

    close = df["Close"].dropna()
    if close.empty:
        return pd.Series(dtype=float)

    last_price = float(close.iloc[-1])
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) > 90:
        returns = returns[-90:]
    mu = returns.mean()

    last_date = df.index[-1]
    freq = pd.infer_freq(df.index)
    future_dates = []
    if freq and "W" in freq.upper():
        for i in range(1, horizon + 1):
            future_dates.append(last_date + pd.Timedelta(days=7 * i))
    else:
        for i in range(1, horizon + 1):
            future_dates.append(last_date + pd.Timedelta(days=i))

    forecast_prices = [last_price * np.exp(mu * i) for i in range(1, horizon + 1)]
    return pd.Series(data=forecast_prices, index=pd.to_datetime(future_dates))


# ============================================================
# STREAMLIT APP
# ============================================================
def main():
    st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")

    st.sidebar.title("ℹ️ App Info")
    st.sidebar.markdown(
        """
        **Stock Technical Analyzer**

        **FinRL Reference**
        - https://medium.com/@zeyneptufekci.etu/reinforcement-learning-in-stock-prediction-with-finrl-8df9769b60dd

        **Disclaimer**
        Educational use only. Not financial advice.
        """
    )

    st.title("📈 Stock Technical Analysis & Recommendation")

    col1, col2, col3 = st.columns([2, 1, 1])

    # ----------------------------------------------------------------
    # ADDITIVE: easier ticker selection via 2 tabs (US vs NSE)
    # - Does NOT change any indicator/recommendation logic below.
    # - Manual ticker input still exists for power users.
    # ----------------------------------------------------------------
    picked_ticker = _ticker_picker_ui()

    # Manual ticker override (kept from your existing app)
    default_ticker = picked_ticker if picked_ticker else "AAPL"
    ticker = col1.text_input("Ticker (manual override)", default_ticker).upper().strip()
    period = col2.selectbox("Period", ["1y", "5y", "max"], index=0)
    interval_map = {"Daily": "1d", "Weekly": "1wk"}
    interval_label = col3.selectbox("Interval", list(interval_map.keys()), index=0)
    interval = interval_map[interval_label]

    st.markdown("---")

    if not ticker:
        st.info("Enter a ticker to begin.")
        return

    df = fetch_stock_data(ticker, period, interval)
    if df.empty:
        st.error("Invalid ticker or no data available for the selected period/interval.")
        return

    df = add_technical_indicators(df)

    st.subheader(f"{ticker} – Latest Indicator Snapshot")
    st.dataframe(df.tail(1))

    # ADDITIVE: Piotroski F-Score (fundamentals) for the selected ticker
    p_score, p_details = compute_piotroski_f_score(ticker)
    if p_score is None:
        st.info("Piotroski F-Score: Not available for this ticker (missing fundamentals).")
    else:
        st.caption(f"Piotroski F-Score (0–9): **{int(p_score)}**  (higher is generally stronger fundamentals)")

    # -----------------------------
    # RENKO SETTINGS (moved here so the same settings drive BOTH:
    #  - Renko chart BUY/SELL/HOLD label
    #  - Indicator Signal Breakdown Renko row
    # (No change to any existing feature/logic; only shared inputs.)
    # -----------------------------
    st.subheader("Renko Settings (used for Renko chart + breakdown)")
    with st.expander("Renko settings", expanded=False):
        st.caption("Default brick size uses **ATR(5)** (5-day volatility). You can change ATR days or override with a fixed brick size.")
        renko_atr_days = st.number_input(
            "ATR days for Renko brick size",
            min_value=1, max_value=60, value=5, step=1,
            key="renko_atr_days"
        )
        use_manual = st.toggle(
            "Override brick size manually",
            value=False,
            key="renko_use_manual"
        )
        manual_brick = st.number_input(
            "Manual brick size (price units)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.4f",
            disabled=not use_manual,
            key="renko_manual_brick",
            help="If enabled, this value is used directly as the brick size.",
        )

    renko_brick_size = _compute_renko_brick_size(
        df,
        atr_days=int(renko_atr_days),
        manual_brick=(float(manual_brick) if use_manual else None),
    )
    renko_df = build_renko(df, brick_size=renko_brick_size)
    renko_rec, renko_reason, renko_score = chart_rec_renko(renko_df, lookback=5)

    # Global recommendation (unchanged)
    signals, score = compute_indicator_signals(df)
    rec, expl = derive_final_recommendation(score)

    if rec == "BUY":
        st.success(f"Recommendation: **{rec}**")
    elif rec == "SELL":
        st.error(f"Recommendation: **{rec}**")
    else:
        st.warning(f"Recommendation: **{rec}**")
    st.write(expl)

    st.subheader("Indicator Signal Breakdown")
    sig_table = pd.DataFrame([{"Indicator": k, "Score": v["score"]} for k, v in signals.items()])
    # Add Renko row into the Indicator Signal Breakdown (display-only; global score unchanged)
    sig_table = pd.concat(
        [sig_table, pd.DataFrame([{"Indicator": "Renko", "Score": int(renko_score)}])],
        ignore_index=True
    )
    st.dataframe(sig_table)

    st.markdown("---")
    st.header("📊 Charts")

    # 1) Price + EMA 20/34/50
    st.subheader("Price + EMA 20 / 34 / 50")
    ema_rec, ema_reason = chart_rec_ema(df)
    label_box(ema_rec, ema_reason)
    st.line_chart(df[["Close", "EMA_20", "EMA_34", "EMA_50"]])

    # 2) Bollinger Bands (with price)
    st.subheader("Bollinger Bands (with Price)")
    bb_rec, bb_reason = chart_rec_bollinger(df)
    label_box(bb_rec, bb_reason)
    st.line_chart(df[["Close", "BB_upper", "BB_lower"]])

    # 3) VWAP with Price (line) + Volume (bar) => price visible
    st.subheader("VWAP with Price + Volume")
    vwap_rec, vwap_reason = chart_rec_vwap(df)
    label_box(vwap_rec, vwap_reason)
    st.line_chart(df[["Close", "VWAP"]])
    st.caption("Volume (separate scale):")
    st.bar_chart(df[["Volume"]])

    # 4) MVWAP with Price (line) + Volume (bar) => price visible
    st.subheader("MVWAP (20) with Price + Volume")
    mvwap_rec, mvwap_reason = chart_rec_mvwap(df)
    label_box(mvwap_rec, mvwap_reason)
    st.line_chart(df[["Close", "MVWAP_20"]])
    st.caption("Volume (separate scale):")
    st.bar_chart(df[["Volume"]])

    # MACD
    st.subheader("MACD (12, 26, 9)")
    macd_rec, macd_reason = chart_rec_macd(df)
    label_box(macd_rec, macd_reason)
    st.line_chart(df[["MACD", "MACD_signal"]])
    st.bar_chart(df[["MACD_hist"]])

    # RSI
    st.subheader("RSI (14)")
    rsi_rec, rsi_reason = chart_rec_rsi(df)
    label_box(rsi_rec, rsi_reason)
    st.line_chart(df[["RSI_14"]])

    # Stochastic
    st.subheader("Stochastic Oscillator (14, 3)")
    stoch_rec, stoch_reason = chart_rec_stoch(df)
    label_box(stoch_rec, stoch_reason)
    st.line_chart(df[["STOCH_K", "STOCH_D"]])

    # ATR
    st.subheader("ATR (14)")
    atr_rec, atr_reason = chart_rec_atr(df)
    label_box(atr_rec, atr_reason)
    st.line_chart(df[["ATR_14"]])

    # Renko
    st.subheader("Renko Chart (ATR-based Brick Size)")

    st.caption(f"Renko brick size in use: **{renko_brick_size:.4f}**")

    if renko_df.empty:
        st.info("Renko: No bricks formed for this period (try a smaller brick size / fewer ATR days, or use a longer period).")
    else:
        label_box(renko_rec, renko_reason)
        fig = plot_renko_bricks(renko_df, renko_brick_size)
        st.plotly_chart(fig, use_container_width=True)

        # Direction as +1/-1 bars (keeps the existing helper view)
        st.caption("Brick direction (+1 up / -1 down):")
        st.bar_chart(renko_df.set_index("BrickIndex")[["Direction"]])


        with st.expander("Renko bricks table", expanded=False):
            st.dataframe(renko_df.tail(200), use_container_width=True)

    # Forecast
    st.markdown("---")
    st.header("🔮 30-Day Price Forecast (FinRL-Inspired)")
    forecast_series = generate_30d_forecast(df, horizon=30)
    if forecast_series.empty:
        st.info("Not enough data to generate a 30-day forecast.")
    else:
        combined = pd.concat([df["Close"], forecast_series.rename("Forecast_Close")], axis=0)
        st.line_chart(combined.to_frame())

    render_market_picks_ui()

    st.caption(f"Data updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
