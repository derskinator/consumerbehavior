
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pytrends.request import TrendReq
import requests
from io import StringIO
from datetime import datetime, timedelta

st.set_page_config(page_title="US Buyer Behavior Dashboard", page_icon="📈", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
# -----------------------------
# External API Connectors (FRED)
# -----------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_fred_series(series_id, api_key=None, start_date=None):
    """
    Fetch a FRED time series as a DataFrame with columns: Date, <series_id>
    Requires an API key (free from fred.stlouisfed.org). If not provided, returns empty df.
    """
    import requests, pandas as pd
    if not api_key:
        return pd.DataFrame()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = str(start_date)
    url = "https://api.stlouisfed.org/fred/series/observations"
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        return pd.DataFrame()
    data = resp.json().get("observations", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)[["date","value"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Convert "value" to numeric, coerce missing '.'
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df = df.rename(columns={"date":"Date", "value": series_id})
    return df.dropna()

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_google_trends(keywords, geo="US", timeframe="today 3-m"):
    if not keywords:
        return pd.DataFrame()
    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time().reset_index()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df = df.rename(columns={"date": "Date"})
    # Normalize each keyword to 0-100 over selected range for comparability
    for k in keywords:
        if k in df.columns and df[k].max() > 0:
            df[k] = (df[k] / df[k].max()) * 100
    return df

def normalize_series(s):
    # Min-max to [0,100]; handle constant series
    if s.max() == s.min():
        return pd.Series([50.0]*len(s), index=s.index, name=s.name)
    return 100 * (s - s.min()) / (s.max() - s.min())

def compute_composite_index(dfs, weights=None):
    """
    Combine multiple time-indexed numeric dataframes/series by aligned dates.
    dfs: dict {name: dataframe with 'Date' column and numeric columns}
    weights: dict {name: weight}; default equal
    Returns long-form and wide-form composite index (0-100).
    """
    pieces = []
    for name, df in dfs.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        if "Date" not in tmp.columns:
            continue
        tmp = tmp.set_index("Date")
        # numeric only
        tmp = tmp.select_dtypes(include=[np.number])
        # Normalize each column then average to a single series for that dataset
        if tmp.shape[1] == 0:
            continue
        tmp_norm = tmp.apply(normalize_series)
        series = tmp_norm.mean(axis=1).rename(name)
        pieces.append(series)

    if not pieces:
        return pd.DataFrame(), pd.DataFrame()

    wide = pd.concat(pieces, axis=1).sort_index()
    # fill gaps by forward fill
    wide = wide.fillna(method="ffill")
    if weights is None:
        weights = {k: 1.0 for k in wide.columns}
    # align weights
    w = pd.Series(weights).reindex(wide.columns).fillna(0.0)
    # Weighted composite 0-100
    composite = (wide.mul(w, axis=1).sum(axis=1) / max(w.sum(), 1e-9)).rename("Composite Index")
    out = pd.concat([wide, composite], axis=1).reset_index().rename(columns={"index": "Date"})
    return out, wide

def detect_anomalies(series, window=14, z_thresh=2.0):
    s = series.copy().dropna()
    if len(s) < window + 2:
        return pd.Series([False]*len(series), index=series.index)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std().replace(0, np.nan)
    zscore = (s - rolling_mean) / rolling_std
    flags = (np.abs(zscore) >= z_thresh)
    return flags.reindex(series.index).fillna(False)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Controls")
st.sidebar.caption("Configure signals and blending")

st.sidebar.subheader("Google Trends (Daily)")
kw_input = st.sidebar.text_input("Enter up to 5 keywords (comma-separated)", value="buy now pay later, coupons, grocery delivery")
timeframe = st.sidebar.selectbox("Timeframe", ["now 7-d", "today 1-m", "today 3-m", "today 12-m"], index=2)
geo = st.sidebar.text_input("Geo (ISO-2 or region code)", value="US")

st.sidebar.subheader("Weights")
w_trends = st.sidebar.slider("Weight: Google Trends", 0.0, 1.0, 0.4, 0.05)
w_weekly = st.sidebar.slider("Weight: Weekly Uploads", 0.0, 1.0, 0.4, 0.05)
w_monthly = st.sidebar.slider("Weight: Monthly Uploads", 0.0, 1.0, 0.2, 0.05)

st.sidebar.subheader("Anomaly Detection")

st.sidebar.subheader("Visualization")
smooth = st.sidebar.checkbox("7-day smoothing (moving average)", value=True)
show_pct_change = st.sidebar.checkbox("Show % change panels", value=True)
show_heatmap = st.sidebar.checkbox("Show keyword heatmap (weekly)", value=True)

z_thresh = st.sidebar.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.1)
window = st.sidebar.slider("Rolling window (days)", 7, 30, 14, 1)

st.sidebar.markdown("---")

st.sidebar.subheader("FRED API (Macro/Weekly)")
fred_key = st.sidebar.text_input("FRED API Key (optional)", value="", type="password")
fred_series_choices = {
    "Retail & Food Services Sales (Monthly, SAAR)": "RSAFS",
    "Univ. of Michigan Consumer Sentiment (Monthly)": "UMCSENT",
    "Initial Unemployment Claims (Weekly)": "ICSA",
    "Personal Consumption Expenditures, Nominal (Monthly)": "PCE",
    "Gasoline Prices, All Grades (Weekly)": "GASREGW",
}
fred_selected = st.sidebar.multiselect("Add FRED series to Monthly/Weekly layers", list(fred_series_choices.keys()), default=["Retail & Food Services Sales (Monthly, SAAR)", "Univ. of Michigan Consumer Sentiment (Monthly)"])

st.sidebar.subheader("Keyword Packs (Google Trends)")
kw_packs = {
    "Deal-Seeking": ["coupons","promo code","discount","clearance","best deals"],
    "Essential Spend": ["grocery delivery","Walmart","Costco","Aldi","food stamps"],
    "Discretionary": ["concert tickets","luxury bags","makeup","video game console","gym membership"],
    "Durables & Home": ["mattress","refrigerator","sofa","home improvement","air conditioner"],
    "Travel & Leisure": ["flights","hotels","airbnb","car rental","cruise"],
    "Financial Stress": ["payday loan","buy now pay later","credit card debt","layaway","debt consolidation"],
    "Back-to-School/Holiday (Seasonal)": ["back to school","Halloween costumes","Black Friday","Cyber Monday","Christmas gifts"]
}
selected_pack = st.sidebar.selectbox("Load a keyword pack (optional)", ["None"] + list(kw_packs.keys()))
if selected_pack != "None":
    kw_input = ", ".join(kw_packs[selected_pack])

st.sidebar.markdown("---")
st.sidebar.caption("Weekly & monthly CSV formats are provided on the home screen.")

# -----------------------------
# Main Layout
# -----------------------------
st.title("📈 U.S. Buyer Behavior — Real‑Time Dashboard (Streamlit)")
st.write(
    "Layer daily search interest, weekly spending signals, and monthly anchors into a single **Composite Index**. "
    "Upload your own weekly/monthly datasets to blend with Google Trends."
)

# Cheat-sheet download links
with st.expander("CSV Templates (click to expand)"):
    st.markdown("**Weekly Template Columns:** Date, Category, Value")
    st.markdown("**Monthly Template Columns:** Date, Category, Value")
    st.download_button("Download weekly_template.csv", data='Date,Category,Value\n2025-08-18,Card Spend,100\n2025-08-25,Card Spend,102\n', file_name="weekly_template.csv")
    st.download_button("Download monthly_template.csv", data='Date,Category,Value\n2025-06-01,Retail Sales,100\n2025-07-01,Retail Sales,101\n', file_name="monthly_template.csv")

# -----------------------------
# Section 1: Daily Google Trends
# -----------------------------
st.header("Daily Signals — Google Trends")
keywords = [k.strip() for k in kw_input.split(",") if k.strip()][:5]
daily_df = fetch_google_trends(keywords=keywords, geo=geo, timeframe=timeframe)

if not daily_df.empty:
    st.line_chart(daily_df.set_index("Date")[keywords])
    st.caption("Normalized to each keyword's 0–100 range for the selected timeframe.")

# Enhanced visualization for Google Trends
if not daily_df.empty and keywords:
    trends = daily_df.set_index("Date")[keywords].copy()
    trends = trends.sort_index()
    if smooth:
        trends_ma = trends.rolling(7, min_periods=1).mean()
        st.write("**Daily Trends (7-day smoothed)**" if smooth else "**Daily Trends**")
        st.plotly_chart(px.line(trends_ma, labels={"value":"Index","Date":"Date","variable":"Keyword"}, title="Google Trends — Smoothed"), use_container_width=True)
    else:
        st.plotly_chart(px.line(trends, labels={"value":"Index","Date":"Date","variable":"Keyword"}, title="Google Trends — Raw"), use_container_width=True)

    # % change panels (WoW and DoD)
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

    # Top movers: last 7d avg vs prior 7d avg
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
    except Exception as e:
        st.caption("Top Movers table unavailable for current date range.")

    # Heatmap (weekly)
    if show_heatmap:
        weekly = trends.resample("W").mean()
        heat = weekly.T  # keywords as rows
        fig = px.imshow(heat, aspect="auto", labels=dict(x="Week", y="Keyword", color="Index"), title="Keyword Heatmap (Weekly Averages)")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter keywords to see Google Trends.")


# -----------------------------
# Section 1.5: FRED Series
# -----------------------------
st.header("Official Indicators — FRED (Optional)")
fred_frames_weekly = []
fred_frames_monthly = []

if fred_selected and fred_key:
    for label in fred_selected:
        sid = fred_series_choices[label]
        fred_df = fetch_fred_series(sid, api_key=fred_key)
        if fred_df.empty:
            st.info(f"FRED series {label} not available (check API key or series).")
            continue
        # Try to infer frequency: weekly series often have frequent dates (ICSA, GASREGW)
        # We'll push weekly series to weekly_df; monthly to monthly_df
        # Heuristic: median delta <= 10 days => weekly-ish
        f = fred_df.copy()
        f["Date"] = pd.to_datetime(f["Date"]).dt.date
        if f["Date"].nunique() >= 3:
            deltas = pd.Series(sorted(f["Date"])) .diff().dropna().astype("timedelta64[D]").dropna()
            is_weekly = (deltas.median() <= 10) if not deltas.empty else False
        else:
            is_weekly = False
        # Pivot like uploads: Date, Category, Value
        cat_name = label
        piv = f.rename(columns={sid: "Value"})
        piv["Category"] = cat_name
        piv = piv[["Date","Category","Value"]]
        if is_weekly:
            fred_frames_weekly.append(piv)
        else:
            fred_frames_monthly.append(piv)

# Merge FRED into weekly and monthly pivots before charts
if fred_frames_weekly:
    fred_wide = pd.concat(fred_frames_weekly).pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
    if weekly_frames:
        weekly_df = pd.merge(weekly_df, fred_wide, on="Date", how="outer")
    else:
        weekly_df = fred_wide

if fred_frames_monthly:
    fred_wide_m = pd.concat(fred_frames_monthly).pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
    if monthly_df.empty:
        monthly_df = fred_wide_m
    else:
        monthly_df = pd.merge(monthly_df, fred_wide_m, on="Date", how="outer")

# -----------------------------
# Section 2: Weekly Signals (Upload)
# -----------------------------
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
    weekly_df = weekly_df.sort_values("Date")
    st.bar_chart(weekly_df.set_index("Date"))

# -----------------------------
# Section 3: Monthly Anchors (Upload)
# -----------------------------
st.header("Monthly Anchors — Upload CSV")
m_file = st.file_uploader("Upload a MONTHLY CSV file (Date, Category, Value)", type=["csv"], key="monthly")

monthly_df = pd.DataFrame()
if m_file:
    m = pd.read_csv(m_file)
    if {"Date","Category","Value"}.issubset(m.columns):
        m["Date"] = pd.to_datetime(m["Date"]).dt.date
        monthly_df = m.pivot_table(index="Date", columns="Category", values="Value", aggfunc="mean").reset_index()
        st.line_chart(monthly_df.set_index("Date"))
    else:
        st.warning("Monthly CSV is missing required columns.")

# -----------------------------
# Composite Index
# -----------------------------
st.header("Composite Buyer Behavior Index")
sources = {}

if not daily_df.empty:
    sources["Daily (Trends)"] = daily_df

if not weekly_df.empty:
    sources["Weekly Uploads"] = weekly_df

if not monthly_df.empty:
    sources["Monthly Uploads"] = monthly_df

weights = {"Daily (Trends)": w_trends, "Weekly Uploads": w_weekly, "Monthly Uploads": w_monthly}

comp_long, comp_wide = compute_composite_index(sources, weights=weights)

if not comp_long.empty:
    st.line_chart(comp_long.set_index("Date")[["Composite Index"]])
    st.caption("Weighted blend of normalized sources. Adjust weights in the sidebar.")

    # Export
    csv = comp_long.to_csv(index=False)
    st.download_button("Download Composite Index (CSV)", data=csv, file_name="composite_index.csv")


    # Anomaly flags on Composite
    comp_long["Date"] = pd.to_datetime(comp_long["Date"])
    comp_long = comp_long.sort_values("Date")
    flags = detect_anomalies(comp_long.set_index("Date")["Composite Index"], window=window, z_thresh=z_thresh)
    flagged_dates = comp_long.loc[flags.values, "Date"].dt.date.tolist()
    if flagged_dates:
        st.warning(f"Anomalies detected (z≥{z_thresh}): {', '.join(map(str, flagged_dates))}")
    else:
        st.success("No composite anomalies flagged given current settings.")
else:
    st.info("Provide at least one data source to compute the composite index.")

# -----------------------------
# Notes
# -----------------------------
with st.expander("Notes & Tips"):
    st.markdown("""
- **Daily:** Use broad consumer-interest terms (e.g., *coupons*, *buy now pay later*, *best deals*, *near me* patterns).
- **Weekly:** Upload card-spend indices (by category), Adobe Digital Economy weekly, or foot traffic series.
- **Monthly:** Upload Census retail sales or BEA PCE categories.
- **Weights:** Start with 0.4 / 0.4 / 0.2. Increase weekly/monthly weights when fresh uploads are added.
- **Anomalies:** Flags show short-term spikes vs trend; confirm with weekly/monthly anchors.
""")
