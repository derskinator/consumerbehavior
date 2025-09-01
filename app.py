import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pytrends.request import TrendReq
from datetime import datetime, timedelta

st.set_page_config(page_title="US Consumer Behavior Dashboard", page_icon="📈", layout="wide")

# ========= Safe defaults =========
SAFE_MODE_DEFAULT = True           # prevents any network calls on first load
GT_REQ_TIMEOUT = 5                 # seconds for pytrends HTTP requests
FRED_REQ_TIMEOUT = 5               # seconds for FRED HTTP requests

# ========= Helpers =========
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_google_trends(keywords, geo="US", timeframe="now 7-d", timeout=GT_REQ_TIMEOUT):
    """Fast, resilient Google Trends fetch with strict timeouts."""
    if not keywords:
        return pd.DataFrame()
    try:
        pytrends = TrendReq(
            hl="en-US",
            tz=360,
            retries=0,
            backoff_factor=0.05,
            requests_args={"timeout": timeout},
        )
        pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time().reset_index()
        if df.empty:
            return df
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        df = df.rename(columns={"date": "Date"})
        for k in keywords:
            if k in df.columns and df[k].max() > 0:
                df[k] = (df[k] / df[k].max()) * 100.0
        return df
    except Exception as e:
        st.warning(f"Google Trends request failed quickly: {e}")
        return pd.DataFrame()

def normalize_series(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series([50.0]*len(s), index=s.index, name=s.name)
    return 100.0 * (s - s.min()) / (s.max() - s.min())

def compute_composite_index(dfs: dict, weights: dict | None = None):
    pieces = []
    for name, df in dfs.items():
        if df is None or df.empty or "Date" not in df.columns:
            continue
        tmp = df.copy().set_index("Date").sort_index()
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
    out = pd.concat([wide, composite], axis=1).reset_index().rename(columns={"index":"Date"})
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

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_fred_series(series_id, api_key=None, start_date=None, timeout=FRED_REQ_TIMEOUT):
    """Fetch a FRED series with a hard timeout; empty if key missing or fails."""
    import requests
    if not api_key:
        return pd.DataFrame()
    try:
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
        if start_date:
            params["observation_start"] = str(start_date)
        url = "https://api.stlouisfed.org/fred/series/observations"
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("observations", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)[["date","value"]]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
        return df.rename(columns={"date":"Date", "value": series_id}).dropna()
    except Exception as e:
        st.warning(f"FRED request failed quickly: {e}")
        return pd.DataFrame()

# ========= Sidebar =========
st.sidebar.title("Controls")
safe_mode = st.sidebar.checkbox("Safe Mode (no external calls until clicked)", value=SAFE_MODE_DEFAULT)
st.sidebar.caption("If startup ever hangs, keep Safe Mode on and use the Load button below.")

st.sidebar.subheader("Google Trends")
kw_input = st.sidebar.text_input("Up to 5 keywords", value="buy now pay later, coupons, grocery delivery")
timeframe = st.sidebar.selectbox("Timeframe", ["now 7-d", "today 1-m", "today 3-m", "today 12-m"], index=0)
geo = st.sidebar.text_input("Geo", value="US")

st.sidebar.subheader("FRED (optional)")
fred_key = st.sidebar.text_input("FRED API Key", value="", type="password")
fred_series_choices = {
    "Retail & Food Services Sales (Monthly, SAAR)": "RSAFS",
    "Univ. of Michigan Consumer Sentiment (Monthly)": "UMCSENT",
    "Initial Unemployment Claims (Weekly)": "ICSA",
    "Personal Consumption Expenditures, Nominal (Monthly)": "PCE",
    "Gasoline Prices, All Grades (Weekly)": "GASREGW",
}
fred_selected = st.sidebar.multiselect("Add FRED series", list(fred_series_choices.keys()), default=[])

st.sidebar.subheader("Weights")
w_trends = st.sidebar.slider("Daily (Trends)", 0.0, 1.0, 0.4, 0.05)
w_weekly = st.sidebar.slider("Weekly Uploads", 0.0, 1.0, 0.4, 0.05)
w_monthly = st.sidebar.slider("Monthly Uploads", 0.0, 1.0, 0.2, 0.05)

st.sidebar.subheader("Anomalies")
z_thresh = st.sidebar.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.1)
window = st.sidebar.slider("Rolling window (days)", 7, 30, 14, 1)

st.sidebar.markdown("---")
run_fetch = st.sidebar.button("▶️ Load / Refresh Data")
st.sidebar.caption("Tip: Keep keywords ≤3 and timeframe = now 7-d for fastest loads.")

# ========= Main =========
st.title("📈 U.S. Consumer Behavior — Real-Time Dashboard")
st.write("Starts in **Safe Mode** so the app never stalls at startup. Click **Load / Refresh Data** when ready.")

with st.expander("CSV Templates"):
    st.markdown("**Weekly CSV** → `Date,Category,Value`")
    st.download_button("Download weekly_template.csv",
        data='Date,Category,Value\n2025-08-18,Card Spend,100\n2025-08-25,Card Spend,102\n',
        file_name="weekly_template.csv")
    st.markdown("**Monthly CSV** → `Date,Category,Value`")
    st.download_button("Download monthly_template.csv",
        data='Date,Category,Value\n2025-06-01,Retail Sales,100\n2025-07-01,Retail Sales,101\n',
        file_name="monthly_template.csv")

# ---- Daily: Google Trends ----
st.header("Daily Signals — Google Trends")
keywords = [k.strip() for k in kw_input.split(",") if k.strip()][:5]
daily_df = pd.DataFrame()
if run_fetch and not safe_mode and keywords:
    daily_df = fetch_google_trends(keywords=keywords, geo=geo, timeframe=timeframe)

if not daily_df.empty:
    st.caption("Normalized to each keyword's 0–100 range for the selected timeframe.")
    st.line_chart(daily_df.set_index("Date")[keywords])
else:
    st.info("Safe Mode is ON or no data loaded. Click **Load / Refresh** to fetch Trends.")

# ---- FRED ----
st.header("Official Indicators — FRED (Optional)")
fred_frames_weekly, fred_frames_monthly = [], []
if run_fetch and not safe_mode and fred_selected and fred_key:
    for label in fred_selected:
        sid = fred_series_choices[label]
        fdf = fetch_fred_series(sid, api_key=fred_key)
        if fdf.empty: 
            continue
        fdf["Date"] = pd.to_datetime(fdf["Date"]).dt.date
        # crude frequency check
        is_weekly = False
        if fdf["Date"].nunique() >= 4:
            dsorted = pd.Series(sorted(fdf["Date"]))
            deltas = dsorted.diff().dropna().astype("timedelta64[D]")
            is_weekly = (deltas.median() <= 10) if not deltas.empty else False
        piv = fdf.rename(columns={sid:"Value"})
        piv["Category"] = label
        piv = piv[["Date","Category","Value"]]
        (fred_frames_weekly if is_weekly else fred_frames_monthly).append(piv)
else:
    st.caption("Provide a FRED key and turn Safe Mode off to load official indicators.")

# ---- Weekly Uploads ----
st.header("Weekly Signals — Upload CSV")
w_files = st.file_uploader("Upload one or more WEEKLY CSV files (Date, Category, Value)", accept_multiple_files=True, type=["csv"], key="weekly")
weekly_frames = []
if w_files:
    for f in w_files:
        df = pd.read_csv(f)
        if {"Date","Category","Value"}.issubset(df.columns):
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            pivot = df.pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
            weekly_frames.append(pivot)
        else:
            st.warning(f"{f.name}: Missing required columns.")

weekly_df = pd.DataFrame()
if weekly_frames:
    weekly_df = weekly_frames[0]
    for extra in weekly_frames[1:]:
        weekly_df = pd.merge(weekly_df, extra, on="Date", how="outer")

    if fred_frames_weekly:
        fred_wide = pd.concat(fred_frames_weekly).pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
        weekly_df = pd.merge(weekly_df, fred_wide, on="Date", how="outer")

    weekly_df = weekly_df.sort_values("Date")
    st.bar_chart(weekly_df.set_index("Date"))
else:
    if fred_frames_weekly:
        fred_wide = pd.concat(fred_frames_weekly).pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
        st.bar_chart(fred_wide.set_index("Date"))

# ---- Monthly Uploads ----
st.header("Monthly Anchors — Upload CSV")
m_file = st.file_uploader("Upload a MONTHLY CSV file (Date, Category, Value)", type=["csv"], key="monthly")
monthly_df = pd.DataFrame()
if m_file:
    m = pd.read_csv(m_file)
    if {"Date","Category","Value"}.issubset(m.columns):
        m["Date"] = pd.to_datetime(m["Date"]).dt.date
        monthly_df = m.pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
    else:
        st.warning("Monthly CSV missing required columns.")
if fred_frames_monthly:
    fred_wide_m = pd.concat(fred_frames_monthly).pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
    monthly_df = fred_wide_m if monthly_df.empty else pd.merge(monthly_df, fred_wide_m, on="Date", how="outer")
if not monthly_df.empty:
    st.line_chart(monthly_df.set_index("Date"))

# ---- Composite ----
st.header("Composite Consumer Behavior Index")
sources = {}
if not daily_df.empty:
    sources["Daily (Trends)"] = daily_df
if not weekly_df.empty:
    sources["Weekly Uploads"] = weekly_df
elif fred_frames_weekly:
    sources["Weekly Uploads"] = pd.concat(fred_frames_weekly).pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
if not monthly_df.empty:
    sources["Monthly Uploads"] = monthly_df

weights = {"Daily (Trends)": w_trends, "Weekly Uploads": w_weekly, "Monthly Uploads": w_monthly}
comp_long, comp_wide = compute_composite_index(sources, weights=weights)
if not comp_long.empty:
    st.line_chart(comp_long.set_index("Date")[["Composite Index"]])
    comp_long["Date"] = pd.to_datetime(comp_long["Date"])
    flags = detect_anomalies(comp_long.set_index("Date")["Composite Index"], window=window, z_thresh=z_thresh)
    flagged_dates = comp_long.loc[flags.values, "Date"].dt.date.tolist()
    if flagged_dates:
        st.warning(f"Anomalies (z≥{z_thresh}): {', '.join(map(str, flagged_dates))}")
    st.download_button("Download Composite Index (CSV)", data=comp_long.to_csv(index=False), file_name="composite_index.csv")
else:
    st.info("No composite yet. Load at least one layer with Safe Mode off.")

# ---- Diagnostics ----
with st.expander("Diagnostics"):
    st.write("**Versions**")
    st.code(f"""
Python runtime OK
pandas {pd.__version__}
numpy {np.__version__}
streamlit {st.__version__}
""")
    import importlib
    try:
        import requests
        st.write("**Internet test**")
        try:
            r = requests.get("https://api.stlouisfed.org", timeout=3)
            st.success(f"FRED reachable: {r.status_code}")
        except Exception as e:
            st.warning(f"FRED reachability failed: {e}")
    except Exception as e:
        st.warning(f"requests import failed: {e}")

    st.write("**Pytrends handshake**")
    if st.button("Test pytrends (2s timeout)"):
        try:
            t = TrendReq(requests_args={"timeout": 2})
            t.build_payload(["test"], timeframe="now 7-d", geo="US")
            st.success("Pytrends handshake OK")
        except Exception as e:
            st.error(f"Pytrends failed quickly: {e}")

st.caption("If the app ever feels stuck, keep Safe Mode on, click Load/Refresh once, and add APIs/plots incrementally.")

