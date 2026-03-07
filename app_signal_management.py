import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Signal Trending Analytics", layout="wide")
st.title("Signal Trending Analytics Dashboard")

# =========================================================
# Helpers
# =========================================================
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


def normalize_yes_no(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "yes": "Yes",
                "y": "Yes",
                "true": "Yes",
                "1": "Yes",
                "no": "No",
                "n": "No",
                "false": "No",
                "0": "No",
            }
        )
    )


# =========================================================
# Upload
# =========================================================
uploaded = st.file_uploader("Upload Line Listing (Excel)", type=["xlsx"])

if uploaded is None:
    st.info("Upload an Excel line listing to begin.")
    st.stop()

df = pd.read_excel(uploaded)

st.subheader("Preview Data")
st.dataframe(df.head(20))

# =========================================================
# Detect columns
# =========================================================
PT_CANDS = ["event pt", "preferred term", "reaction pt", "meddra pt", "pt"]
DATE_CANDS = ["receipt date", "case receipt date", "initial receipt date", "date"]
SERIOUS_CANDS = ["serious", "serious case flag", "case level seriousness"]
FATAL_CANDS = ["fatal", "death flag", "fatal case flag"]
PRODUCT_CANDS = ["suspect product", "suspect product name", "product", "drug", "generic name"]
COUNTRY_CANDS = ["country", "primary source country", "country of occurrence"]
SOC_CANDS = ["soc", "event soc", "ae system organ class", "system organ class"]
AGE_CANDS = ["age"]
GENDER_CANDS = ["gender", "sex"]
OUTCOME_CANDS = ["outcome", "event outcome", "case outcome"]
LISTED_CANDS = ["listedness", "expectedness", "case-level expectedness", "event listedness"]

pt_col = find_col(df, PT_CANDS)
date_col = find_col(df, DATE_CANDS)
ser_col = find_col(df, SERIOUS_CANDS)
fat_col = find_col(df, FATAL_CANDS)
prod_col = find_col(df, PRODUCT_CANDS)
country_col = find_col(df, COUNTRY_CANDS)
soc_col = find_col(df, SOC_CANDS)
age_col = find_col(df, AGE_CANDS)
gender_col = find_col(df, GENDER_CANDS)
outcome_col = find_col(df, OUTCOME_CANDS)
listed_col = find_col(df, LISTED_CANDS)

st.subheader("Detected Columns")
st.write(
    {
        "Event PT": pt_col,
        "Date": date_col,
        "Serious": ser_col,
        "Fatal": fat_col,
        "Product": prod_col,
        "Country": country_col,
        "SOC": soc_col,
        "Age": age_col,
        "Gender": gender_col,
        "Outcome": outcome_col,
        "Listedness": listed_col,
    }
)

if pt_col is None or date_col is None:
    st.error("At minimum, Event PT and Date columns must be detected.")
    st.stop()

# =========================================================
# Prepare data
# =========================================================
df = df.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df[df[date_col].notna()].copy()

if df.empty:
    st.error("No valid dates found in the uploaded file.")
    st.stop()

df["Month"] = df[date_col].dt.to_period("M").astype(str)

if ser_col:
    df[ser_col] = normalize_yes_no(df[ser_col])

if fat_col:
    df[fat_col] = normalize_yes_no(df[fat_col])

# Age buckets
if age_col:
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    def age_bucket(x):
        if pd.isna(x):
            return "Not Reported"
        if x < 2:
            return "Infant"
        if x < 12:
            return "Child"
        if x < 18:
            return "Adolescent"
        if x < 55:
            return "Adult (18-54)"
        if x < 65:
            return "Adult (55-64)"
        return "Elderly (65+)"

    df["Age Group"] = df[age_col].apply(age_bucket)

# =========================================================
# Filters
# =========================================================
st.sidebar.header("Filters")

filtered_df = df.copy()

if prod_col:
    product_options = ["All"] + sorted(filtered_df[prod_col].dropna().astype(str).unique().tolist())
    selected_product = st.sidebar.selectbox("Product", product_options)
    if selected_product != "All":
        filtered_df = filtered_df[filtered_df[prod_col].astype(str) == selected_product]

if country_col:
    country_options = ["All"] + sorted(filtered_df[country_col].dropna().astype(str).unique().tolist())
    selected_country = st.sidebar.selectbox("Country", country_options)
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df[country_col].astype(str) == selected_country]

if soc_col:
    soc_options = ["All"] + sorted(filtered_df[soc_col].dropna().astype(str).unique().tolist())
    selected_soc = st.sidebar.selectbox("SOC", soc_options)
    if selected_soc != "All":
        filtered_df = filtered_df[filtered_df[soc_col].astype(str) == selected_soc]

month_options = sorted(filtered_df["Month"].dropna().unique().tolist())
selected_months = st.sidebar.multiselect("Months", month_options, default=month_options)
if selected_months:
    filtered_df = filtered_df[filtered_df["Month"].isin(selected_months)]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

top_n = st.sidebar.slider("Top PTs to display", 5, 25, 10, 10)

# =========================================================
# KPIs
# =========================================================
st.subheader("Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", len(filtered_df))
k2.metric("Unique PTs", filtered_df[pt_col].nunique())
if prod_col:
    k3.metric("Products", filtered_df[prod_col].nunique())
if country_col:
    k4.metric("Countries", filtered_df[country_col].nunique())

# =========================================================
# 1. Case Volume Trend
# =========================================================
monthly_cases = filtered_df.groupby("Month").size().reset_index(name="Case Count")

# =========================================================
# 2. Serious vs Non-serious Trend
# =========================================================
if ser_col:
    serious_trend = (
        filtered_df.groupby(["Month", ser_col])
        .size()
        .reset_index(name="Case Count")
    )
else:
    serious_trend = None

c1, c2 = st.columns(2)

with c1:
    st.subheader("1. Case Volume Trend")
    fig1 = px.bar(
        monthly_cases,
        x="Month",
        y="Case Count",
        title="Monthly Case Volume"
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("2. Serious vs Non-serious Trend")
    if serious_trend is not None:
        fig2 = px.bar(
            serious_trend,
            x="Month",
            y="Case Count",
            color=ser_col,
            barmode="stack",
            title="Monthly Seriousness Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Seriousness column not detected.")

# =========================================================
# 3. Country Distribution
# 9. Outcome Distribution
# =========================================================
c3, c4 = st.columns(2)

with c3:
    st.subheader("3. Country Distribution")
    if country_col:
        country_counts = (
            filtered_df[country_col]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        country_counts.columns = ["Country", "Case Count"]

        fig3 = px.bar(
            country_counts.head(15),
            x="Country",
            y="Case Count",
            title="Top Countries"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Country column not detected.")

with c4:
    st.subheader("9. Outcome Distribution")
    if outcome_col:
        outcome_counts = (
            filtered_df[outcome_col]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        outcome_counts.columns = ["Outcome", "Count"]

        fig9 = px.pie(
            outcome_counts,
            names="Outcome",
            values="Count",
            title="Outcome Distribution"
        )
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("Outcome column not detected.")

# =========================================================
# 4. Top PT Frequency
# =========================================================
pt_counts = (
    filtered_df[pt_col]
    .astype(str)
    .value_counts()
    .head(top_n)
    .reset_index()
)
pt_counts.columns = ["Event PT", "Case Count"]

# =========================================================
# 5. SOC Distribution
# =========================================================
c5, c6 = st.columns(2)

with c5:
    st.subheader("4. Top PT Frequency")
    fig4 = px.bar(
        pt_counts.sort_values("Case Count", ascending=True),
        x="Case Count",
        y="Event PT",
        orientation="h",
        title="Top PTs"
    )
    st.plotly_chart(fig4, use_container_width=True)

with c6:
    st.subheader("5. SOC Distribution")
    if soc_col:
        soc_counts = (
            filtered_df[soc_col]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        soc_counts.columns = ["SOC", "Count"]

        fig5 = px.treemap(
            soc_counts,
            path=["SOC"],
            values="Count",
            title="SOC Distribution"
        )
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("SOC column not detected.")

# =========================================================
# 6. PT Trend Over Time
# =========================================================
st.subheader("6. PT Trend Over Time")

pt_options = sorted(filtered_df[pt_col].dropna().astype(str).unique().tolist())
selected_pt = st.selectbox("Select PT for trend view", pt_options)

selected_pt_df = filtered_df[filtered_df[pt_col].astype(str) == selected_pt]
selected_pt_trend = (
    selected_pt_df.groupby("Month")
    .size()
    .reset_index(name="Case Count")
)

fig6 = px.line(
    selected_pt_trend,
    x="Month",
    y="Case Count",
    markers=True,
    title=f"Monthly Trend for {selected_pt}"
)
st.plotly_chart(fig6, use_container_width=True)

# =========================================================
# 7. Spike Detection Table
# =========================================================
st.subheader("7. Spike Detection Table")

monthly_pt_counts = (
    filtered_df.groupby(["Month", pt_col])
    .size()
    .reset_index(name="Case Count")
)

pivot = monthly_pt_counts.pivot_table(
    index=pt_col,
    columns="Month",
    values="Case Count",
    fill_value=0
)

spike_table = None

if pivot.shape[1] >= 2:
    prev_col = pivot.columns[-2]
    curr_col = pivot.columns[-1]

    spike_table = pivot.copy()
    spike_table["Previous Month"] = spike_table[prev_col]
    spike_table["Current Month"] = spike_table[curr_col]
    spike_table["Absolute Increase"] = spike_table["Current Month"] - spike_table["Previous Month"]
    spike_table["Relative Increase %"] = np.where(
        spike_table["Previous Month"] == 0,
        np.nan,
        ((spike_table["Current Month"] - spike_table["Previous Month"]) / spike_table["Previous Month"]) * 100
    )

    spike_table = spike_table.reset_index()
    spike_table = spike_table.rename(columns={pt_col: "Event PT"})
    spike_table = spike_table[
        ["Event PT", "Previous Month", "Current Month", "Absolute Increase", "Relative Increase %"]
    ].sort_values(["Absolute Increase", "Relative Increase %"], ascending=False)

    st.dataframe(spike_table.head(25), use_container_width=True)
else:
    st.info("At least 2 months of data are needed for spike detection.")

# =========================================================
# 8. Demographics (Age / Gender)
# =========================================================
st.subheader("8. Demographics")

d1, d2 = st.columns(2)

with d1:
    if age_col:
        age_counts = (
            filtered_df["Age Group"]
            .value_counts()
            .reset_index()
        )
        age_counts.columns = ["Age Group", "Count"]

        age_order = [
            "Infant",
            "Child",
            "Adolescent",
            "Adult (18-54)",
            "Adult (55-64)",
            "Elderly (65+)",
            "Not Reported",
        ]
        age_counts["Age Group"] = pd.Categorical(age_counts["Age Group"], categories=age_order, ordered=True)
        age_counts = age_counts.sort_values("Age Group")

        fig8a = px.bar(
            age_counts,
            x="Age Group",
            y="Count",
            title="Age Group Distribution"
        )
        st.plotly_chart(fig8a, use_container_width=True)
    else:
        st.info("Age column not detected.")

with d2:
    if gender_col:
        gender_counts = (
            filtered_df[gender_col]
            .astype(str)
            .value_counts()
            .reset_index()
        )
        gender_counts.columns = ["Gender", "Count"]

        fig8b = px.pie(
            gender_counts,
            names="Gender",
            values="Count",
            title="Gender Distribution"
        )
        st.plotly_chart(fig8b, use_container_width=True)
    else:
        st.info("Gender column not detected.")

# =========================================================
# Fatal trend bonus
# =========================================================
if fat_col:
    st.subheader("Fatal Case Trend")
    fatal_df = filtered_df[filtered_df[fat_col] == "Yes"]
    if not fatal_df.empty:
        fatal_trend = fatal_df.groupby("Month").size().reset_index(name="Fatal Cases")
        fig_fatal = px.bar(
            fatal_trend,
            x="Month",
            y="Fatal Cases",
            title="Monthly Fatal Case Trend"
        )
        st.plotly_chart(fig_fatal, use_container_width=True)
    else:
        st.info("No fatal cases in filtered dataset.")

# =========================================================
# Download
# =========================================================
st.subheader("Download Trend Tables")

download_df = monthly_pt_counts.copy()
csv = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Monthly PT Trend Table",
    data=csv,
    file_name="monthly_pt_trend_table.csv",
    mime="text/csv",
)

output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    filtered_df.to_excel(writer, index=False, sheet_name="Filtered Raw Data")
    monthly_pt_counts.to_excel(writer, index=False, sheet_name="Monthly PT Trends")
    pt_counts.to_excel(writer, index=False, sheet_name="Top PT Frequency")
    if spike_table is not None:
        spike_table.to_excel(writer, index=False, sheet_name="Spike Detection")

st.download_button(
    "Download Full Excel Report",
    data=output.getvalue(),
    file_name="Signal_Trending_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
