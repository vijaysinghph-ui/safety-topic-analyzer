import os
import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="Safety Topic Analyzer")
st.title("Safety Topic Analyzer")

# ---------- OpenAI Client ----------
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------- Helper Functions ----------
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


def yes_count(series):
    s = series.astype(str).str.strip().str.lower()
    return int(s.isin(["y", "yes", "true", "1"]).sum())


def top_values(series, n=5):
    return series.dropna().astype(str).value_counts().head(n).to_dict()


# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Excel line listing", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head(20))

    # ---- Column Candidates ----
    PT_CANDIDATES = ["event pt", "preferred term", "meddra pt", "reaction pt", "pt"]
    CASE_CANDIDATES = ["case number", "case id", "icsr", "report id"]

    SERIOUS_CANDS = ["serious case flag", "serious", "event seriousness"]
    FATAL_CANDS = ["death flag", "fatal case flag", "fatal"]
    LISTED_CANDS = ["listedness", "event listedness", "expected"]
    DECH_CANDS = ["dechallenge", "dechallenge results"]
    RECH_CANDS = ["rechallenge", "rechallenge results"]
    ONSET_CANDS = ["onset latency", "time to onset", "event start date"]
    COUNTRY_CANDS = ["country"]
    NARR_CANDS = ["narrative"]

    # ---- Detect Columns ----
    pt_col = find_col(df, PT_CANDIDATES)
    case_col = find_col(df, CASE_CANDIDATES)

    ser_col = find_col(df, SERIOUS_CANDS)
    fat_col = find_col(df, FATAL_CANDS)
    list_col = find_col(df, LISTED_CANDS)
    dech_col = find_col(df, DECH_CANDS)
    rech_col = find_col(df, RECH_CANDS)
    onset_col = find_col(df, ONSET_CANDS)
    country_col = find_col(df, COUNTRY_CANDS)
    narr_col = find_col(df, NARR_CANDS)

    st.write("Detected Columns:")
    st.write({
        "Event PT": pt_col,
        "Case ID": case_col,
        "Serious": ser_col,
        "Fatal": fat_col,
        "Listedness": list_col,
        "Dechallenge": dech_col,
        "Rechallenge": rech_col,
        "Onset": onset_col,
        "Country": country_col,
        "Narrative": narr_col,
    })

    if not pt_col:
        st.error("Event PT column not detected. Please rename to include 'Event PT' or 'Preferred Term'.")
        st.stop()

    # ---- Group by PT ----
    grouped = df.groupby(pt_col, dropna=True)
    topic_table = grouped.size().reset_index(name="case_count")
    topic_table = topic_table.sort_values("case_count", ascending=False)

    st.subheader("Auto-grouped topics (by Event PT)")
    st.dataframe(topic_table.head(50))

    # ---------- Mode Selection ----------
    st.subheader("Choose Analysis Mode")

    mode = st.radio(
        "Select Mode",
        ["Single PT Deep Dive (PBRER / Signal Assessment)",
         "Bulk Trending (Monthly Scan)"]
    )

    def build_evidence(subset, pt_value):
        return {
            "event_pt": str(pt_value),
            "total_cases": int(len(subset)),
            "serious_yes": yes_count(subset[ser_col]) if ser_col else None,
            "fatal_yes": yes_count(subset[fat_col]) if fat_col else None,
            "listedness": top_values(subset[list_col]) if list_col else None,
            "dechallenge": top_values(subset[dech_col]) if dech_col else None,
            "rechallenge": top_values(subset[rech_col]) if rech_col else None,
            "onset_examples": subset[onset_col].dropna().astype(str).head(5).tolist() if onset_col else [],
            "countries": top_values(subset[country_col]) if country_col else None,
            "narratives": subset[narr_col].dropna().astype(str).head(2).tolist() if narr_col else [],
        }

    # ==========================
    # MODE A: Single PT
    # ==========================
    if mode.startswith("Single"):

        pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

        if st.button("Analyze Selected PT"):
            subset = df[df[pt_col] == pt_choice]
            evidence = build_evidence(subset, pt_choice)

            prompt = f"""
You are a senior pharmacovigilance physician preparing a PBRER safety topic evaluation.

Provide:
1) Event PT
2) Safety topic decision: Include / Monitor / Not a priority
3) Case synopsis (max 6 bullets)
4) Bradford Hill assessment (brief but medical):
   - Strength
   - Consistency
   - Temporality
   - Biological gradient
   - Plausibility
   - Coherence
   - Experiment
   - Analogy
5) Causality conclusion
6) PBRER-ready summary (150 words max)
7) Recommended next action

Evidence:
{evidence}
"""

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
            )

            st.subheader(f"Deep Dive – {pt_choice}")
            st.write(response.choices[0].message.content)

    # ==========================
    # MODE B: Bulk Trending
    # ==========================
    else:

        TOP_N = st.slider("Number of PTs to Analyze", 3, 30, 10)

        if st.button("Analyze Top PTs"):
            top_pts = topic_table.head(TOP_N)[pt_col].tolist()

            for pt in top_pts:
                subset = df[df[pt_col] == pt]
                evidence = build_evidence(subset, pt)

                prompt = f"""
You are a senior pharmacovigilance physician performing monthly safety trending.

Provide SHORT output:
1) Event PT
2) Safety topic decision
3) Key evidence (max 4 bullets)
4) Bradford Hill headline
5) Causality conclusion
6) One-paragraph summary (100 words max)
7) Recommended action

Evidence:
{evidence}
"""

                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": prompt}],
                )

                st.subheader(f"Trending – {pt}")
                st.write(response.choices[0].message.content)
                st.markdown("---")
