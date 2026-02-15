import os
import io
import json
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
# OpenAI Client
# ----------------------------
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    api_key = None

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
        # exact match
        for i, c in enumerate(cols_l):
            if c == cand_l:
                return cols[i]
        # contains match
        for i, c in enumerate(cols_l):
            if cand_l in c:
                return cols[i]
    return None

def yes_count(series):
    s = series.astype(str).str.strip().str.lower()
    return int(s.isin(["y", "yes", "true", "1"]).sum())

def top_values(series, n=5):
    return series.dropna().astype(str).value_counts().head(n).to_dict()

def looks_like_bullets(text: str) -> bool:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return False
    bulletish = 0
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            bulletish += 1
        elif len(ln) >= 2 and ln[0].isdigit() and ln[1] in [")", ".", ":"]:
            bulletish += 1
    return bulletish >= max(2, int(0.4 * len(lines)))

def ensure_paragraph(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            ln = ln.lstrip("-•* ").strip()
        cleaned.append(ln)
    para = " ".join(cleaned)
    while "  " in para:
        para = para.replace("  ", " ")
    return para.strip()

def json_or_none(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

# ----------------------------
# Upload Excel
# ----------------------------
uploaded_file = st.file_uploader("Upload Excel Line Listing", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(20))

    # Column candidates
    PT_CANDIDATES = ["event pt", "preferred term", "meddra pt", "reaction pt", "pt"]
    SERIOUS_CANDS = ["serious case flag", "serious", "event seriousness"]
    FATAL_CANDS = ["death flag", "fatal case flag", "fatal"]
    LISTED_CANDS = ["listedness", "event listedness", "expected"]
    DECH_CANDS = ["dechallenge", "dechallenge results"]
    RECH_CANDS = ["rechallenge", "rechallenge results"]
    ONSET_CANDS = ["onset latency", "time to onset", "event start date"]
    COUNTRY_CANDS = ["country"]
    NARR_CANDS = ["narrative"]

    # Detect columns
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
        st.error("Event PT column not detected. Rename column to include 'Event PT' or 'Preferred Term'.")
        st.stop()

    # Group by PT
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
    # SINGLE PT MODE (STABILIZED JSON OUTPUT)
    # ==================================================
    if mode.startswith("Single"):

        pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

        if st.button("Analyze & Generate Word Report"):

            subset = df[df[pt_col] == pt_choice]
            evidence = build_evidence(subset, pt_choice)

            # Force JSON so we control order & formatting
            prompt = f"""
You are a senior pharmacovigilance physician.

Return ONLY valid JSON (no markdown, no extra text). Use this exact schema and keys:

{{
  "Background of Drug-Event Combination": "string",
  "Case Synopsis": "string",
  "Bradford Hill Assessment": "string",
  "Regulatory Landscape Overview": "string",
  "Causality Conclusion": "string",
  "Safety Topic Decision": "string",
  "PBRER-Ready Summary": "string",
  "Recommended Next Action": "string"
}}

STRICT RULES:
1) Background of Drug-Event Combination MUST be product-level background ONLY:
   - pharmacologic class / mechanism relevant to the event
   - known class effects
   - general pregnancy/lactation considerations if broadly known
   - labeling recognition in general terms (if broadly known)
   - DO NOT include any case counts, countries, listedness, seriousness numbers, or reporting interval details.
   - DO NOT mention "France", "Switzerland", "n=...", "%", "cases", "serious", "fatal", "listed/unlisted".

2) Case Synopsis MUST contain ALL reporting interval details:
   - total cases, seriousness, fatality, geography, listedness, timing, confounders, and brief clinical description.

3) Regulatory Landscape Overview:
   - Do NOT fabricate regulatory actions.
   - If uncertain, write exactly:
     "No widely recognized regulatory action specific to this drug-event combination."

4) PBRER-Ready Summary:
   - ONE paragraph only (NO bullets, NO line breaks)
   - 140–180 words
   - Integrate evidence from Case Synopsis + Bradford Hill + Causality Conclusion + Safety Topic Decision
   - Do NOT add new facts

EVIDENCE (reporting interval):
{evidence}
"""

            resp = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content
            data = json_or_none(raw)

            # One repair attempt if JSON failed
            if data is None:
                repair = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": "Convert the following into ONLY valid JSON matching the exact schema. Output JSON only.\n\n" + raw}],
                )
                data = json_or_none(repair.choices[0].message.content)

            if data is None:
                st.error("Model output was not valid JSON. Showing raw output for debugging:")
                st.code(raw)
                st.stop()

            # Enforce paragraph summary even if model slips
            summary = str(data.get("PBRER-Ready Summary", "")).strip()
            if looks_like_bullets(summary) or "\n" in summary:
                summary_fixed = ensure_paragraph(summary)

                # If still not good, do a micro-fix call
                fix_prompt = f"""
Rewrite into a single regulatory narrative paragraph (NO bullets, NO line breaks), 140–180 words.
Do not add new facts. End with benefit–risk position.

TEXT:
{summary}
"""
                fix = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": fix_prompt}],
                )
                summary_fixed = ensure_paragraph(fix.choices[0].message.content)
                data["PBRER-Ready Summary"] = summary_fixed

            ORDER = [
                "Background of Drug-Event Combination",
                "Case Synopsis",
                "Bradford Hill Assessment",
                "Regulatory Landscape Overview",
                "Causality Conclusion",
                "Safety Topic Decision",
                "PBRER-Ready Summary",
                "Recommended Next Action",
            ]

            # Display in app in fixed order
            st.subheader(f"Deep Dive – {pt_choice}")
            for k in ORDER:
                st.markdown(f"### {k}")
                st.write(str(data.get(k, "")).strip())

            # ---------------- Word Export ----------------
            document = Document()
            document.add_heading("Safety Topic Evaluation Report", level=1)
            document.add_paragraph(f"Event PT: {pt_choice}")
            document.add_paragraph("")

            document.add_heading("Evaluation", level=2)

            for k in ORDER:
                document.add_heading(k, level=3)
                val = str(data.get(k, "")).strip()
                if k == "PBRER-Ready Summary":
                    val = ensure_paragraph(val)
                    document.add_paragraph(val)
                else:
                    parts = [p for p in val.split("\n") if p.strip()]
                    if not parts:
                        document.add_paragraph("")
                    else:
                        for p in parts:
                            document.add_paragraph(p)

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
