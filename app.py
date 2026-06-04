"""
HR Attrition Dashboard — Streamlit
Run locally:  streamlit run app.py
Deploy:       push this folder (with train.csv + test.csv) to GitHub,
              then create an app on https://share.streamlit.io pointing at app.py.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------- CONFIG
st.set_page_config(page_title="HR Attrition Dashboard",
                   page_icon="📊", layout="wide")

LEFT_COLOR, STAY_COLOR, TEMPLATE = "#E45756", "#4C78A8", "plotly_white"

ORDINAL_ORDERS = {
    "work_life_balance":    ["Poor", "Below Average", "Fair", "Good", "Excellent"],
    "job_satisfaction":     ["Very Low", "Low", "Medium", "High", "Very High"],
    "performance_rating":   ["Low", "Below Average", "Average", "Good", "High"],
    "education_level":      ["High School", "Associate", "Bachelor's", "Master's", "PhD"],
    "job_level":            ["Entry", "Mid", "Senior"],
    "company_size":         ["Small", "Medium", "Large"],
    "company_reputation":   ["Very Poor", "Poor", "Fair", "Good", "Excellent"],
    "employee_recognition": ["Very Low", "Low", "Medium", "High", "Very High"],
}


# --------------------------------------------------------------------- DATA
def _normalize_columns(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))
    return df


def _clean(df):
    df = df.copy()
    target_map = {"left": 1, "yes": 1, "1": 1, "1.0": 1,
                  "stayed": 0, "no": 0, "0": 0, "0.0": 0}
    df["attrition"] = (df["attrition"].astype(str).str.strip().str.lower()
                       .map(target_map).astype("Int64"))
    df = df.drop(columns=[c for c in ["employee_id"] if c in df.columns])
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    for col, order in ORDINAL_ORDERS.items():
        if col in df.columns:
            present = [v for v in order if v in df[col].unique()]
            df[col] = pd.Categorical(df[col], categories=present, ordered=True)
    return df.drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner="Loading data…", ttl=3600)
def load_data():
    """Read train.csv + test.csv from the archive subfolder, combine, clean."""
    here = Path(__file__).resolve().parent
    archive = here / "archive"
    frames = []
    for name in ["train.csv", "test.csv"]:
        p = archive / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        st.error(f"Looked in: `{archive}` — files not found.")
        return None
    df = _normalize_columns(pd.concat(frames, ignore_index=True))
    return _clean(df)


def attrition_rate_by(df, col):
    g = (df.groupby(col, observed=True)["attrition"]
         .agg(rate="mean", n="size").reset_index())
    g["rate"] = (g["rate"] * 100).round(1)
    return g


# --------------------------------------------------------------------- LOAD
df = load_data()

st.title("📊 Employee Attrition Dashboard")
st.caption("Synthetic HR dataset · explore who leaves and why. "
           "Use the filters on the left to slice the workforce.")

if df is None:
    st.error("Could not find **train.csv** / **test.csv** next to `app.py`. "
             "Add them to the repository (or upload below) and reload.")
    up = st.file_uploader("Upload train.csv and test.csv", type="csv",
                          accept_multiple_files=True)
    if up:
        raw = _normalize_columns(pd.concat([pd.read_csv(f) for f in up],
                                           ignore_index=True))
        df = _clean(raw)
    else:
        st.stop()

baseline_rate = df["attrition"].mean() * 100

# --------------------------------------------------------------------- SIDEBAR
st.sidebar.header("Filters")


def multiselect_filter(frame, col, label):
    if col not in frame.columns:
        return frame
    opts = [o for o in (ORDINAL_ORDERS.get(col) or sorted(frame[col].dropna().unique()))
            if o in frame[col].unique()]
    chosen = st.sidebar.multiselect(label, opts, default=opts)
    return frame[frame[col].isin(chosen)] if chosen else frame


f = df.copy()
f = multiselect_filter(f, "job_role", "Job Role")
f = multiselect_filter(f, "job_level", "Job Level")
f = multiselect_filter(f, "company_size", "Company Size")
f = multiselect_filter(f, "gender", "Gender")
f = multiselect_filter(f, "remote_work", "Remote Work")

if "age" in f.columns:
    lo, hi = int(df["age"].min()), int(df["age"].max())
    a0, a1 = st.sidebar.slider("Age range", lo, hi, (lo, hi))
    f = f[f["age"].between(a0, a1)]

if "monthly_income" in f.columns:
    lo, hi = int(df["monthly_income"].min()), int(df["monthly_income"].max())
    i0, i1 = st.sidebar.slider("Monthly income ($)", lo, hi, (lo, hi), step=100)
    f = f[f["monthly_income"].between(i0, i1)]

st.sidebar.markdown(f"**{len(f):,}** of {len(df):,} employees selected")

if f.empty:
    st.warning("No employees match these filters. Widen the selection.")
    st.stop()

# --------------------------------------------------------------------- KPIs
left = int(f["attrition"].sum())
rate = f["attrition"].mean() * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees (filtered)", f"{len(f):,}")
c2.metric("Left", f"{left:,}")
c3.metric("Attrition Rate", f"{rate:.1f}%",
          delta=f"{rate - baseline_rate:+.1f} pts vs company",
          delta_color="inverse")
if "monthly_income" in f.columns:
    c4.metric("Avg Monthly Income", f"${f['monthly_income'].mean():,.0f}")

st.divider()

# --------------------------------------------------------------------- CHARTS
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "By Category", "Numeric Drivers", "Data"])

with tab1:
    a, b = st.columns([1, 2])
    with a:
        fig = go.Figure(go.Pie(labels=["Stayed", "Left"],
                               values=[len(f) - left, left], hole=0.55,
                               marker_colors=[STAY_COLOR, LEFT_COLOR],
                               textinfo="label+percent"))
        fig.update_layout(title="Stayed vs Left", template=TEMPLATE,
                          showlegend=False, height=380,
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        if "job_role" in f.columns:
            d = attrition_rate_by(f, "job_role").sort_values("rate")
            fig = px.bar(d, x="rate", y="job_role", orientation="h",
                         text="rate", color="rate",
                         color_continuous_scale="Reds",
                         labels={"rate": "Attrition rate (%)", "job_role": ""})
            fig.update_traces(texttemplate="%{text:.1f}%")
            fig.add_vline(x=baseline_rate, line_dash="dash", line_color="gray",
                          annotation_text=f"Avg {baseline_rate:.1f}%")
            fig.update_layout(title="Attrition Rate by Job Role",
                              template=TEMPLATE, coloraxis_showscale=False,
                              height=380, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    cats = [c for c in ["work_life_balance", "job_satisfaction",
                        "company_reputation", "employee_recognition",
                        "marital_status", "education_level"] if c in f.columns]
    pick = st.selectbox("Break attrition down by:", cats,
                        format_func=lambda s: s.replace("_", " ").title())
    d = attrition_rate_by(f, pick)
    fig = px.bar(d, x=d[pick].astype(str), y="rate", text="rate",
                 color="rate", color_continuous_scale="Reds",
                 labels={"rate": "Attrition rate (%)", "x": ""})
    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.add_hline(y=baseline_rate, line_dash="dash", line_color="gray",
                  annotation_text=f"Company avg {baseline_rate:.1f}%")
    fig.update_layout(title=f"Attrition Rate by {pick.replace('_', ' ').title()}",
                      template=TEMPLATE, coloraxis_showscale=False, height=460)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    num = [c for c in ["monthly_income", "age", "years_at_company",
                       "company_tenure", "distance_from_home",
                       "number_of_promotions"] if c in f.columns]
    pick = st.selectbox("Compare stayers vs leavers on:", num,
                        format_func=lambda s: s.replace("_", " ").title())
    plot_df = f.assign(status=f["attrition"].map({0: "Stayed", 1: "Left"}))
    fig = px.box(plot_df, x="status", y=pick, color="status",
                 color_discrete_map={"Stayed": STAY_COLOR, "Left": LEFT_COLOR},
                 labels={pick: pick.replace("_", " ").title(), "status": ""})
    fig.update_layout(title=f"{pick.replace('_', ' ').title()} by Attrition Status",
                      template=TEMPLATE, showlegend=False, height=460)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.dataframe(f.head(500), use_container_width=True, height=420)
    st.download_button("⬇️ Download filtered data (CSV)",
                       f.to_csv(index=False).encode(),
                       file_name="filtered_attrition.csv", mime="text/csv")

st.caption("Synthetic data — findings are illustrative, not real-world HR truths.")
