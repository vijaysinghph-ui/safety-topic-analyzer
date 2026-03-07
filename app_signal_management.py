import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="PV Signal Trending Analytics", layout="wide")

st.title("Pharmacovigilance Signal Trending Analytics")

# ---------------------------
# Column detection helper
# ---------------------------
def find_col(df, candidates):

    cols = list(df.columns)
    cols_l = [str(c).lower().strip() for c in cols]

    for cand in candidates:

        cand_l = cand.lower()

        for i, c in enumerate(cols_l):
            if c == cand_l:
                return cols[i]

        for i, c in enumerate(cols_l):
            if cand_l in c:
                return cols[i]

    return None


# ---------------------------
# Upload File
# ---------------------------
uploaded = st.file_uploader("Upload Line Listing", type=["xlsx"])

if uploaded is None:
    st.info("Upload Excel file to begin.")
    st.stop()

df = pd.read_excel(uploaded)

st.subheader("Preview Data")
st.dataframe(df.head(20))

# ---------------------------
# Detect columns
# ---------------------------
PT_CANDS = ["event pt", "preferred term", "reaction pt", "pt"]
DATE_CANDS = ["receipt date", "case receipt date", "date"]
SERIOUS_CANDS = ["serious", "serious case flag"]
FATAL_CANDS = ["fatal", "death flag"]
PRODUCT_CANDS = ["suspect product", "product", "drug"]

pt_col = find_col(df, PT_CANDS)
date_col = find_col(df, DATE_CANDS)
ser_col = find_col(df, SERIOUS_CANDS)
fat_col = find_col(df, FATAL_CANDS)
prod_col = find_col(df, PRODUCT_CANDS)

st.subheader("Detected Columns")

st.write({
    "Event PT": pt_col,
    "Date": date_col,
    "Serious": ser_col,
    "Fatal": fat_col,
    "Product": prod_col
})

if pt_col is None or date_col is None:
    st.error("Event PT and Date columns are required.")
    st.stop()

# ---------------------------
# Prepare Date
# ---------------------------
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df["Month"] = df[date_col].dt.to_period("M").astype(str)

# ---------------------------
# Product Filter
# ---------------------------
if prod_col:

    product = st.selectbox(
        "Select Product",
        ["All"] + list(df[prod_col].dropna().unique())
    )

    if product != "All":
        df = df[df[prod_col] == product]

# ---------------------------
# Monthly PT counts
# ---------------------------
monthly_counts = (
    df.groupby(["Month", pt_col])
    .size()
    .reset_index(name="Case Count")
)

# ---------------------------
# Top PTs
# ---------------------------
top_n = st.slider("Select Top PTs", 5, 25, 10)

top_pts = (
    df[pt_col]
    .value_counts()
    .head(top_n)
    .index
)

trend_df = monthly_counts[monthly_counts[pt_col].isin(top_pts)]

# ---------------------------
# Trend Chart
# ---------------------------
st.subheader("Monthly PT Trends")

fig = px.line(
    trend_df,
    x="Month",
    y="Case Count",
    color=pt_col,
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Serious Trend
# ---------------------------
if ser_col:

    serious_df = df[df[ser_col].astype(str).str.lower().isin(["yes","y","true","1"])]

    serious_counts = (
        serious_df.groupby(["Month", pt_col])
        .size()
        .reset_index(name="Serious Cases")
    )

    st.subheader("Serious Case Trends")

    fig2 = px.line(
        serious_counts,
        x="Month",
        y="Serious Cases",
        color=pt_col
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Fatal Trend
# ---------------------------
if fat_col:

    fatal_df = df[df[fat_col].astype(str).str.lower().isin(["yes","y","true","1"])]

    fatal_counts = (
        fatal_df.groupby(["Month", pt_col])
        .size()
        .reset_index(name="Fatal Cases")
    )

    st.subheader("Fatal Case Trends")

    fig3 = px.line(
        fatal_counts,
        x="Month",
        y="Fatal Cases",
        color=pt_col
    )

    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# Spike Detection
# ---------------------------
st.subheader("PT Spike Detection")

pivot = monthly_counts.pivot_table(
    index=pt_col,
    columns="Month",
    values="Case Count",
    fill_value=0
)

if pivot.shape[1] >= 2:

    pivot["Previous Month"] = pivot.iloc[:, -2]
    pivot["Current Month"] = pivot.iloc[:, -1]

    pivot["Spike Ratio"] = (
        pivot["Current Month"] /
        pivot["Previous Month"].replace(0, np.nan)
    )

    spikes = pivot.sort_values("Spike Ratio", ascending=False)

    st.dataframe(spikes.head(20))

# ---------------------------
# Internal PRR Approximation
# ---------------------------
st.subheader("Internal Signal Indicator (PRR-like)")

total_cases = len(df)

prr_table = []

for pt in df[pt_col].unique():

    a = len(df[df[pt_col] == pt])
    b = total_cases - a

    prr = (a / total_cases)

    prr_table.append({
        "Event PT": pt,
        "Case Count": a,
        "Relative Frequency": prr
    })

prr_df = pd.DataFrame(prr_table)

st.dataframe(
    prr_df.sort_values(
        "Relative Frequency",
        ascending=False
    ).head(20)
)

# ---------------------------
# Heatmap
# ---------------------------
st.subheader("PT Activity Heatmap")

heat = monthly_counts.pivot_table(
    index=pt_col,
    columns="Month",
    values="Case Count",
    fill_value=0
)

fig4 = px.imshow(
    heat,
    aspect="auto",
    labels=dict(color="Cases")
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# Signal Prioritization Table
# ---------------------------
st.subheader("Signal Prioritization")

priority = spikes.reset_index()

priority["Priority Score"] = (
    priority["Current Month"] * priority["Spike Ratio"]
)

priority = priority.sort_values(
    "Priority Score",
    ascending=False
)

st.dataframe(priority.head(20))

# ---------------------------
# Download Table
# ---------------------------
csv = priority.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Signal Priority Table",
    data=csv,
    file_name="signal_priority_table.csv",
    mime="text/csv",
)
