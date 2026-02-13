import os
import io
import streamlit as st
import pandas as pd
from openai import OpenAI
from docx import Document

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Safety Topic Analyzer", layout="wide")
st.title("Safety Topic Analyzer")

# ----------------------------
# Session State Storage
# ----------------------------
if "results" not in st.session_state:
    st.session_state["results"] = []

# ----------------------------
# OpenAI Client
# ----------------------------
api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------
# Helper Functions
# ----------------------------
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

# ----------------------------
# Upload Excel
# ----------------------------
uploaded_file = st.file_uploader("Upload Excel Line Listing", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(20))

    # Column detection
    PT_CANDIDATES = ["event pt", "preferred term", "meddra pt", "reaction pt", "pt"]
    SERIOUS_CANDS = ["serious case flag", "serious", "event seriousness"]
    FATAL_CANDS = ["death flag", "fatal case flag", "fatal"]
    LISTED_CANDS = ["listedness", "event listedness", "expected"]
    DECH_CANDS = ["dechallenge", "dechallenge results"]
    RECH_CANDS = ["rechallenge", "rechallenge results"]
    ONSET_CANDS = ["onset latency", "time to onset", "event start date"]
    COUNTRY_CANDS = ["country"]
    NARR_CANDS = ["narrative"]

    pt_col = find_col(df, PT_CANDIDATES)
    ser_col = find_col(df, SERIOUS_CANDS)
    fat_col = find_col(df, FATAL_CANDS)
    list_col = find_col(df, LISTED_CANDS)
    dech_col = find_col(df, DECH_CANDS)
    rech_col = find_col(df, RECH_CANDS)
    onset_col = find_col(df, ONSET_CANDS)
    country_col = find_col(df, COUNTRY_CANDS)
    narr_col = find_col(df, NARR_CANDS)

    if not pt_col:
        st.error("Event PT column not detected.")
        st.stop()

    topic_table = (
        df.groupby(pt_col, dropna=True)
        .size()
        .reset_index(name="case_count")
        .sort_values("case_count", ascending=False)
    )

    st.subheader("Auto-grouped Topics (by Event PT)")
    st.dataframe(topic_table.head(50))

    # ----------------------------
    # Mode Selection
    # ----------------------------
    mode = st.radio(
        "Choose Mode",
        ["Single PT Deep Dive (PBRER / Signal Assessment)",
         "Bulk Trending (Monthly Scan)"]
    )

    def build_evidence(subset, pt_value):
        return {
            "event_pt": str(pt_value),
            "total_cases": int(len(subset)),
            "serious_cases": yes_count(subset[ser_col]) if ser_col else None,
            "fatal_cases": yes_count(subset[fat_col]) if fat_col else None,
            "listedness": top_values(subset[list_col]) if list_col else None,
            "dechallenge": top_values(subset[dech_col]) if dech_col else None,
            "rechallenge": top_values(subset[rech_col]) if rech_col else None,
            "onset_examples": subset[onset_col].dropna().astype(str).head(5).tolist() if onset_col else [],
            "countries": top_values(subset[country_col]) if country_col else None,
            "narrative_snippets": subset[narr_col].dropna().astype(str).head(2).tolist() if narr_col else [],
        }

    # ==================================================
    # SINGLE PT MODE
    # ==================================================
    if mode.startswith("Single"):

        pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

        if st.button("Analyze & Generate Word Report"):

            subset = df[df[pt_col] == pt_choice]
            evidence = build_evidence(subset, pt_choice)

            prompt = f"""
You are a senior pharmacovigilance physician preparing a regulatory-grade PBRER safety topic evaluation.

Write in formal regulatory tone. Avoid speculative language.

Provide clearly separated sections:

Safety Topic Decision
Background of Drug-Event Combination
Case Synopsis
Bradford Hill Assessment
Regulatory Landscape Overview
Causality Conclusion
PBRER-Ready Summary
Recommended Next Action

Background of Drug-Event Combination:
- Relevant pharmacology
- Known class effects
- Label recognition status
- Mechanistic plausibility

Regulatory Landscape Overview:
- Known safety communications (FDA, EMA, MHRA)
- Boxed warnings or contraindications (if generally known)
- If uncertain, state: "No widely recognized regulatory action specific to this drug-event combination."

Do NOT fabricate regulatory actions.

Evidence:
{evidence}
"""

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
            )

            output_text = response.choices[0].message.content

            st.subheader(f"Deep Dive – {pt_choice}")
            st.write(output_text)

            # ---------------- Word Export ----------------
            document = Document()
            document.add_heading("Safety Topic Evaluation Report", level=1)
            document.add_paragraph(f"Event PT: {pt_choice}")
            document.add_paragraph("")

            document.add_heading("Evaluation", level=2)

            SECTION_TITLES = [
                "Safety Topic Decision",
                "Background of Drug-Event Combination",
                "Case Synopsis",
                "Bradford Hill Assessment",
                "Regulatory Landscape Overview",
                "Causality Conclusion",
                "PBRER-Ready Summary",
                "Recommended Next Action",
            ]

            lines = [ln.rstrip() for ln in output_text.splitlines()]
            current_title = None
            buffer = []

            def flush_section(title, buf):
                if not title:
                    return
                document.add_heading(title, level=3)
                text = "\n".join([b for b in buf if b.strip() != ""]).strip()
                if text:
                    for para in text.split("\n"):
                        document.add_paragraph(para)

            for ln in lines:
                stripped = ln.strip()
                if stripped in SECTION_TITLES:
                    flush_section(current_title, buffer)
                    current_title = stripped
                    buffer = []
                else:
                    buffer.append(ln)

            flush_section(current_title, buffer)

            document.add_page_break()
            document.add_heading("Data Snapshot", level=2)
            document.add_paragraph(f"Total Cases: {len(subset)}")
            if ser_col:
                document.add_paragraph(f"Serious Cases: {yes_count(subset[ser_col])}")
            if fat_col:
                document.add_paragraph(f"Fatal Cases: {yes_count(subset[fat_col])}")

            doc_buffer = io.BytesIO()
            document.save(doc_buffer)
            doc_buffer.seek(0)

            st.download_button(
                label="Download Word Report",
                data=doc_buffer,
                file_name=f"Safety_Topic_{pt_choice}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    # ==================================================
    # BULK MODE
    # ==================================================
    else:

        TOP_N = st.slider("Number of PTs to Analyze", 3, 30, 10)

        if st.button("Analyze Top PTs"):

            top_pts = topic_table.head(TOP_N)[pt_col].tolist()

            for pt in top_pts:

                subset = df[df[pt_col] == pt]
                evidence = build_evidence(subset, pt)

                prompt = f"""
You are a senior pharmacovigilance physician performing monthly safety trending.

Provide:
1) Event PT
2) Safety topic decision
3) Key evidence (max 4 bullets)
4) Causality conclusion
5) Recommended action

Evidence:
{evidence}
"""

                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": prompt}],
                )

                output_text = response.choices[0].message.content

                st.subheader(f"Trending – {pt}")
                st.write(output_text)
                st.markdown("---")
