import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pytrends.request import TrendReq
from io import StringIO
from datetime import datetime, timedelta

st.set_page_config(page_title="US Consumer Behavior Dashboard", page_icon="📈", layout="wide")

# ======================================
# Helpers
# ======================================
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_google_trends(keywords, geo="US", timeframe="now 7-d"):
    """Fast, resilient Google Trends fetch with normalization."""
    if not keywords:
        return pd.DataFrame()
    try:
        pytrends = TrendReq(
            hl="en-US",
            tz=360,
            retries=1,
            backoff_factor=0.1,
            requests_args={"timeout": 8},
        )
        pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time().reset_index()
        if df.empty:
            return df
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        df = df.rename(columns={"date": "Date"})
        # Normalize each keyword to 0–100
        for k in keywords:
            if k in df.columns and df[k].max() > 0:
                df[k] = (df[k] / df[k].max()) * 100.0
        return df
    except Exception:
        return pd.DataFrame()

def normalize_series(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series([50.0]*len(s), index=s.index, name=s.name)
    return 100.0 * (s - s.min()) / (s.max() - s.min())

def compute_composite_index(dfs: dict, weights: dict | None = None):
    """Blend multiple time-indexed numeric frames/series into a 0–100 composite."""
    pieces = []
    for name, df in dfs.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        if "Date" not in tmp.columns:
            continue
        tmp = tmp.set_index("Date").sort_index()
        tmp = tmp.select_dtypes(include=[np.number])
        if tmp.shape[1] == 0:
            continue
        tmp_norm = tmp.apply(normalize_series)
        series = tmp_norm.mean(axis=1).rename(name)
        pieces.append(series)

    if not pieces:
        return pd.DataFrame(), pd.DataFrame()

    wide = pd.concat(pieces, axis=1).sort_index().ffill()
    if weights is None:
        weights = {k: 1.0 for k in wide.columns}
    w = pd.Series(weights).reindex(wide.columns).fillna(0.0)
    denom = float(w.sum()) if float(w.sum()) > 0 else 1.0
    composite = (wide.mul(w, axis=1).sum(axis=1) / denom).rename("Composite Index")
    out = pd.concat([wide, composite], axis=1).reset_index().rename(columns={"index": "Date"})
    return out, wide

def detect_anomalies(series: pd.Series, window: int = 14, z_thresh: float = 2.0):
    s = series.copy().dropna()
    if len(s) < window + 2:
        return pd.Series([False]*len(series), index=series.index)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std().replace(0, np.nan)
    zscore = (s - rolling_mean) / rolling_std
    flags = (np.abs(zscore) >= z_thresh)
    return flags.reindex(series.index).fillna(False)

# -----------------------------
# External API Connectors (FRED)
# -----------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_fred_series(series_id, api_key=None, start_date=None):
    """
    Fetch a FRED time series as a DataFrame with columns: Date, <series_id>
    Requires an API key. If not provided, returns empty df.
    """
    import requests
    if not api_key:
        return pd.DataFrame()
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start_date:
        params["observation_start"] = str(start_date)
    url = "https://api.stlouisfed.org/fred/series/observations"
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json().get("observations", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)[["date","value"]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
        df = df.rename(columns={"date":"Date", "value": series_id})
        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ======================================
# Sidebar Controls
# ======================================
st.sidebar.title("Controls")
st.sidebar.caption("Configure signals, APIs, and blending")

# Google Trends controls
st.sidebar.subheader("Google Trends (Daily)")
kw_input = st.sidebar.text_input("Enter up to 5 keywords (comma-separated)", value="buy now pay later, coupons, grocery delivery")
timeframe = st.sidebar.selectbox("Timeframe", ["now 7-d", "today 1-m", "today 3-m", "today 12-m"], index=0)
geo = st.sidebar.text_input("Geo (ISO-2 or region code)", value="US")

# Keyword packs
st.sidebar.subheader("Keyword Packs")
kw_packs = {
    "Deal-Seeking": ["coupons","promo code","discount","clearance","best deals"],
    "Essential Spend": ["grocery delivery","Walmart","Costco","Aldi","food stamps"],
    "Discretionary": ["concert tickets","luxury bags","makeup","video game console","gym membership"],
    "Durables & Home": ["mattress","refrigerator","sofa","home improvement","air conditioner"],
    "Travel & Leisure": ["flights","hotels","airbnb","car rental","cruise"],
    "Financial Stress": ["payday loan","buy now pay later","credit card debt","layaway","debt consolidation"],
    "Seasonal": ["back to school","Halloween costumes","Black Friday","Cyber Monday","Christmas gifts"],
}
selected_pack = st.sidebar.selectbox("Load a pack (optional)", ["None"] + list(kw_packs.keys()))
if selected_pack != "None":
    kw_input = ", ".join(kw_packs[selected_pack])

# FRED
st.sidebar.subheader("FRED API (Macro/Weekly)")
fred_key = st.sidebar.text_input("FRED API Key (optional)", value="", type="password")
fred_series_choices = {
    "Retail & Food Services Sales (Monthly, SAAR)": "RSAFS",
    "Univ. of Michigan Consumer Sentiment (Monthly)": "UMCSENT",
    "Initial Unemployment Claims (Weekly)": "ICSA",
    "Personal Consumption Expenditures, Nominal (Monthly)": "PCE",
    "Gasoline Prices, All Grades (Weekly)": "GASREGW",
}
fred_selected = st.sidebar.multiselect(
    "Add FRED series",
    list(fred_series_choices.keys()),
    default=[]
)

# Weights
st.sidebar.subheader("Weights")
w_trends = st.sidebar.slider("Weight: Daily (Trends)", 0.0, 1.0, 0.4, 0.05)
w_weekly = st.sidebar.slider("Weight: Weekly Uploads", 0.0, 1.0, 0.4, 0.05)
w_monthly = st.sidebar.slider("Weight: Monthly Uploads", 0.0, 1.0, 0.2, 0.05)

# Anomaly Detection
st.sidebar.subheader("Anomaly Detection")
z_thresh = st.sidebar.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.1)
window = st.sidebar.slider("Rolling window (days)", 7, 30, 14, 1)

# Visualization toggles (lighter defaults for speed)
st.sidebar.subheader("Visualization")
smooth = st.sidebar.checkbox("7-day smoothing (moving average)", value=True)
show_pct_change = st.sidebar.checkbox("Show % change panels", value=False)
show_heatmap = st.sidebar.checkbox("Show keyword heatmap (weekly)", value=False)

# Manual run button to avoid reruns on every change
st.sidebar.markdown("---")
run_fetch = st.sidebar.button("▶️ Load / Refresh Data")

st.sidebar.markdown("---")
st.sidebar.caption("Upload formats for Weekly/Monthly are described in the app.")

# ======================================
# Main
# ======================================
st.title("📈 U.S. Consumer Behavior — Real-Time Dashboard (Streamlit)")
st.write(
    "Blend **daily Google Trends**, **weekly spending signals**, and **monthly anchors** into a single "
    "**Composite Consumer Behavior Index**. Use the sidebar to pick keyword packs, add FRED series, and adjust weights."
)

with st.expander("CSV Templates (click to expand)"):
    st.markdown("**Weekly Template Columns:** Date, Category, Value")
    st.markdown("**Monthly Template Columns:** Date, Category, Value")
    st.download_button("Download weekly_template.csv", data='Date,Category,Value\n2025-08-18,Card Spend,100\n2025-08-25,Card Spend,102\n', file_name="weekly_template.csv")
    st.download_button("Download monthly_template.csv", data='Date,Category,Value\n2025-06-01,Retail Sales,100\n2025-07-01,Retail Sales,101\n', file_name="monthly_template.csv")

# --------------------------------------
# Section 1: Daily Google Trends
# --------------------------------------
st.header("Daily Signals — Google Trends")
keywords = [k.strip() for k in kw_input.split(",") if k.strip()][:5]

daily_df = pd.DataFrame()
if run_fetch and keywords:
    daily_df = fetch_google_trends(keywords=keywords, geo=geo, timeframe=timeframe)

if not daily_df.empty:
    st.caption("Normalized to each keyword's 0–100 range for the selected timeframe.")
    st.line_chart(daily_df.set_index("Date")[keywords])

    # Enhanced visualization
    trends = daily_df.set_index("Date")[keywords].copy().sort_index()

    # Downsample for very long ranges to keep it snappy
    if len(trends) > 800:
        trends = trends.iloc[::2, :]

    if smooth:
        trends_ma = trends.rolling(7, min_periods=1).mean()
        st.plotly_chart(px.line(trends_ma, labels={"value":"Index","Date":"Date","variable":"Keyword"}, title="Google Trends — Smoothed (7d MA)"), use_container_width=True)
    else:
        st.plotly_chart(px.line(trends, labels={"value":"Index","Date":"Date","variable":"Keyword"}, title="Google Trends — Raw"), use_container_width=True)

    if show_pct_change:
        pct = trends.pct_change().replace([np.inf, -np.inf], np.nan)*100.0
        pct_wow = trends.resample("W").mean().pct_change()*100.0
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Day-over-Day % Change**")
            st.plotly_chart(px.line(pct, labels={"value":"% Δ DoD","Date":"Date","variable":"Keyword"}, title="DoD % Change"), use_container_width=True)
        with col2:
            st.write("**Week-over-Week % Change**")
            st.plotly_chart(px.line(pct_wow, labels={"value":"% Δ WoW","Date":"Date","variable":"Keyword"}, title="WoW % Change"), use_container_width=True)

    # Top movers (7d vs prior 7d)
    try:
        last_date = trends.index.max()
        win = 7
        recent = trends.loc[last_date - pd.Timedelta(days=win-1): last_date].mean()
        prior = trends.loc[last_date - pd.Timedelta(days=2*win-1): last_date - pd.Timedelta(days=win)].mean()
        movers = pd.DataFrame({
            "Keyword": recent.index,
            "Last 7d Avg": recent.values,
            "Prev 7d Avg": prior.values
        })
        movers["Abs Δ"] = movers["Last 7d Avg"] - movers["Prev 7d Avg"]
        movers["% Δ"] = np.where(movers["Prev 7d Avg"]>0, 100*(movers["Abs Δ"]/movers["Prev 7d Avg"]), np.nan)
        movers = movers.sort_values("% Δ", ascending=False)
        st.write("### Top Movers (last 7d vs prior 7d)")
        st.dataframe(movers.style.format({"Last 7d Avg":"{:.1f}", "Prev 7d Avg":"{:.1f}", "Abs Δ":"{:.1f}", "% Δ":"{:.1f}%"}), use_container_width=True)
    except Exception:
        st.caption("Top Movers table unavailable for current date range.")

    if show_heatmap:
        weekly = trends.resample("W").mean()
        heat = weekly.T  # keywords as rows
        fig = px.imshow(heat, aspect="auto", labels=dict(x="Week", y="Keyword", color="Index"), title="Keyword Heatmap (Weekly Averages)")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click **Load / Refresh Data** to fetch Google Trends.")

# --------------------------------------
# Section 1.5: FRED Series (Optional)
# --------------------------------------
st.header("Official Indicators — FRED (Optional)")
fred_frames_weekly = []
fred_frames_monthly = []

if run_fetch and fred_selected and fred_key:
    for label in fred_selected:
        sid = fred_series_choices[label]
        fred_df = fetch_fred_series(sid, api_key=fred_key)
        if fred_df.empty:
            st.info(f"FRED series {label} not available (check API key or series).")
            continue
        f = fred_df.copy()
        f["Date"] = pd.to_datetime(f["Date"]).dt.date
        # Heuristic frequency detection
        if f["Date"].nunique() >= 4:
            dates_sorted = pd.Series(sorted(f["Date"]))
            deltas = dates_sorted.diff().dropna().astype("timedelta64[D]")
            is_weekly = (deltas.median() <= 10) if not deltas.empty else False
        else:
            is_weekly = False
        piv = f.rename(columns={sid: "Value"})
        piv["Category"] = label
        piv = piv[["Date","Category","Value"]]
        if is_weekly:
            fred_frames_weekly.append(piv)
        else:
            fred_frames_monthly.append(piv)
else:

