import os
import io
import json
import re
import streamlit as st
import pandas as pd
from openai import OpenAI
from docx import Document

# =========================================================
# Signal Validation MVP — Stable, Defensible, Pilot-Ready
# Outcomes: Valid Signal / Non-Validated Signal
# =========================================================

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Signal Validation MVP", layout="wide")
st.title("Signal Validation (MVP)")

SYSTEM_STYLE = """
You are an EU QPPV-level safety physician.
You are performing SIGNAL VALIDATION (not signal assessment).
Your writing must be inspection-ready, conservative, and evidence-based.
Do not speculate. Do not recommend risk minimization or label changes.
If data is insufficient, clearly state limitations.
""".strip()

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

def call_openai(user_prompt: str, model: str = "gpt-5") -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_STYLE},
            {"role": "user", "content": user_prompt},
        ],
    )
    return r.choices[0].message.content

# ----------------------------
# Helpers
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

def json_or_none(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def must_have_keys(d, keys):
    return isinstance(d, dict) and all(k in d for k in keys)

def rating_norm(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    if "not" in s and "assess" in s:
        return "Not assessable"
    if "weak" in s:
        return "Weak"
    if "moderate" in s:
        return "Moderate"
    if "strong" in s:
        return "Strong"
    if "low" in s:
        return "Low"
    if "high" in s:
        return "High"
    return s.title()

def ensure_single_paragraph(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if ln.startswith(("-", "•", "*")):
            ln = ln.lstrip("-•* ").strip()
        cleaned.append(ln)
    para = " ".join(cleaned).strip()
    while "  " in para:
        para = para.replace("  ", " ")
    return para

def validation_matrix(overall_woe: str, confounding: str) -> str:
    """
    Deterministic outcome for SIGNAL VALIDATION only:
    - Valid Signal: evidence suggests potential association -> proceed to assessment
    - Non-Validated Signal: evidence insufficient/unsupportive -> no escalation at this time
    """
    overall = rating_norm(overall_woe)
    conf = rating_norm(confounding)

    if overall == "Strong":
        return "Valid Signal"
    if overall == "Moderate":
        if conf in ["Low", "Moderate"]:
            return "Valid Signal"
        return "Non-Validated Signal"
    # Weak / Not assessable / anything else
    return "Non-Validated Signal"

# ----------------------------
# Sidebar: Product Context (stabilizes Background)
# ----------------------------
st.sidebar.header("Product Context (recommended)")
product_name = st.sidebar.text_input("Suspect Product (brand or substance)", value="")
generic_name = st.sidebar.text_input("Generic name (optional)", value="")
ther_class = st.sidebar.text_input("Therapeutic class (optional)", value="")
indication = st.sidebar.text_input("Indication (optional)", value="")

def product_context_string():
    parts = []
    if product_name.strip():
        parts.append(f"Suspect product: {product_name.strip()}")
    if generic_name.strip():
        parts.append(f"Generic name: {generic_name.strip()}")
    if ther_class.strip():
        parts.append(f"Therapeutic class: {ther_class.strip()}")
    if indication.strip():
        parts.append(f"Indication: {indication.strip()}")
    if not parts:
        parts.append("Suspect product: Not provided (keep background high-level and cautious).")
    return "\n".join(parts)

# ----------------------------
# Upload Excel
# ----------------------------
uploaded_file = st.file_uploader("Upload Excel Line Listing", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload an Excel line listing to begin.")
    st.stop()

df = pd.read_excel(uploaded_file)
st.subheader("Preview of Uploaded Data")
st.dataframe(df.head(20))

# ----------------------------
# Column detection
# ----------------------------
PT_CANDS = ["event pt", "preferred term", "meddra pt", "reaction pt", "pt"]
CASE_CANDS = ["case number", "case id", "icsr", "report id", "safety report id"]
SERIOUS_CANDS = ["serious case flag", "serious", "event seriousness", "seriousness"]
FATAL_CANDS = ["death flag", "fatal case flag", "fatal"]
LISTED_CANDS = ["listedness", "event listedness", "expected"]
DECH_CANDS = ["dechallenge", "dechallenge results"]
RECH_CANDS = ["rechallenge", "rechallenge results"]
ONSET_CANDS = ["onset latency", "time to onset", "event start date"]
COUNTRY_CANDS = ["country"]
NARR_CANDS = ["narrative"]
DRUG_CANDS = ["suspect product name", "suspect drug", "drug", "product", "generic name"]
IND_CANDS = ["indication", "indication(s) as reported"]

pt_col = find_col(df, PT_CANDS)
case_col = find_col(df, CASE_CANDS)
ser_col = find_col(df, SERIOUS_CANDS)
fat_col = find_col(df, FATAL_CANDS)
list_col = find_col(df, LISTED_CANDS)
dech_col = find_col(df, DECH_CANDS)
rech_col = find_col(df, RECH_CANDS)
onset_col = find_col(df, ONSET_CANDS)
country_col = find_col(df, COUNTRY_CANDS)
narr_col = find_col(df, NARR_CANDS)
drug_col = find_col(df, DRUG_CANDS)
ind_col = find_col(df, IND_CANDS)

if not pt_col:
    st.error("Event PT column not detected. Rename column to include 'Event PT' or 'Preferred Term'.")
    st.stop()

st.subheader("Detected Columns")
st.write({
    "Event PT": pt_col,
    "Case ID (optional)": case_col,
    "Serious": ser_col,
    "Fatal": fat_col,
    "Listedness": list_col,
    "Dechallenge": dech_col,
    "Rechallenge": rech_col,
    "Onset/Start": onset_col,
    "Country": country_col,
    "Narrative": narr_col,
    "Drug": drug_col,
    "Indication": ind_col,
})

# ----------------------------
# Group by PT
# ----------------------------
topic_table = (
    df.groupby(pt_col, dropna=True)
    .size()
    .reset_index(name="row_count")
    .sort_values("row_count", ascending=False)
)
st.subheader("Auto-grouped Topics (by Event PT)")
st.dataframe(topic_table.head(50))

# ----------------------------
# Select PT for validation
# ----------------------------
st.subheader("Signal Validation Report")
pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

# ----------------------------
# Evidence builder (stable, minimal, defensible)
# ----------------------------
def build_evidence(subset, pt_value):
    e = {
        "event_pt": str(pt_value),
        "row_count": int(len(subset)),
        "unique_case_count": int(subset[case_col].nunique()) if case_col else None,
        "serious_yes": yes_count(subset[ser_col]) if ser_col else None,
        "fatal_yes": yes_count(subset[fat_col]) if fat_col else None,
        "countries_top": top_values(subset[country_col]) if country_col else None,
        "listedness_top": top_values(subset[list_col]) if list_col else None,
        "dechallenge_top": top_values(subset[dech_col]) if dech_col else None,
        "rechallenge_top": top_values(subset[rech_col]) if rech_col else None,
        "onset_examples": subset[onset_col].dropna().astype(str).head(5).tolist() if onset_col else [],
        "indications_top": top_values(subset[ind_col]) if ind_col else None,
        "suspect_products_top": top_values(subset[drug_col]) if drug_col else None,
        "narrative_snippets": subset[narr_col].dropna().astype(str).head(3).tolist() if narr_col else [],
    }
    return e

# ----------------------------
# Generate report
# ----------------------------
if st.button("Generate Signal Validation + Word"):
    subset = df[df[pt_col] == pt_choice]
    evidence = build_evidence(subset, pt_choice)

    # 1) AI returns CRITERIA ratings as JSON (controlled)
    score_keys = [
        "Strength of Evidence",
        "Consistency",
        "Temporality",
        "Biological Plausibility",
        "Confounding Impact",
        "Overall Weight of Evidence",
        "WoE Summary (brief)",
    ]

    scoring_prompt = f"""
Return ONLY valid JSON (no markdown, no extra text) with EXACT keys:

{{
  "Strength of Evidence": "Weak/Moderate/Strong",
  "Consistency": "Weak/Moderate/Strong",
  "Temporality": "Weak/Moderate/Strong/Not assessable",
  "Biological Plausibility": "Weak/Moderate/Strong/Not assessable",
  "Confounding Impact": "Low/Moderate/High/Not assessable",
  "Overall Weight of Evidence": "Weak/Moderate/Strong",
  "WoE Summary (brief)": "2-4 sentences summarizing the weight-of-evidence for signal validation only. No recommendations."
}}

Rules:
- Use only provided evidence; do not speculate.
- If a criterion cannot be assessed, use "Not assessable".
- Keep WoE summary neutral and inspection-ready.

PRODUCT CONTEXT:
{product_context_string()}

EVIDENCE:
{evidence}
"""
    raw_scores = call_openai(scoring_prompt)
    scores = json_or_none(raw_scores)

    if scores is None or not must_have_keys(scores, score_keys):
        repair = call_openai("Convert into ONLY valid JSON with the exact keys. Output JSON only.\n\n" + raw_scores)
        scores = json_or_none(repair)

    if scores is None or not must_have_keys(scores, score_keys):
        st.error("Could not parse scoring JSON. Raw output below:")
        st.code(raw_scores)
        st.stop()

    # Normalize key ratings
    for k in ["Strength of Evidence", "Consistency", "Temporality", "Biological Plausibility", "Confounding Impact", "Overall Weight of Evidence"]:
        scores[k] = rating_norm(scores.get(k, ""))

    # 2) Deterministic outcome from matrix (YOU control this)
    outcome = validation_matrix(scores.get("Overall Weight of Evidence", ""), scores.get("Confounding Impact", ""))

    # 3) Structured narrative sections as JSON (no PBRER)
    section_keys = [
        "Background of Drug-Event Combination",
        "Case Overview",
        "Validation Criteria Evaluation",
        "Signal Weight-of-Evidence (WoE) Summary",
        "Final Validation Outcome",
    ]

    structured_prompt = f"""
Return ONLY valid JSON with EXACT keys and in EXACT order:

{{
  "Background of Drug-Event Combination": "string",
  "Case Overview": "string",
  "Validation Criteria Evaluation": "string",
  "Signal Weight-of-Evidence (WoE) Summary": "string",
  "Final Validation Outcome": "string"
}}

Rules:
- Background: product-level only, <=120 words, ONE paragraph, NO case counts.
- Case Overview: include only reporting-interval facts from evidence (counts, seriousness, geography, listedness, patterns, confounders). No conclusions.
- Validation Criteria Evaluation: short paragraph tying the criteria to the evidence (no new facts).
- WoE Summary: short paragraph consistent with criteria ratings; do not recommend actions.
- Final Validation Outcome MUST be EXACTLY one of:
  "Valid Signal" or "Non-Validated Signal"
  and must be EXACTLY: "{outcome}"

PRODUCT CONTEXT:
{product_context_string()}

CRITERIA RATINGS (fixed; do not change values):
{scores}

EVIDENCE:
{evidence}
"""
    raw_struct = call_openai(structured_prompt)
    report = json_or_none(raw_struct)

    if report is None or not must_have_keys(report, section_keys):
        repair = call_openai("Convert into ONLY valid JSON with the exact keys. Output JSON only.\n\n" + raw_struct)
        report = json_or_none(repair)

    if report is None or not must_have_keys(report, section_keys):
        st.error("Could not parse report JSON. Raw output below:")
        st.code(raw_struct)
        st.stop()

    # enforce formatting
    report["Background of Drug-Event Combination"] = ensure_single_paragraph(report.get("Background of Drug-Event Combination", ""))
    report["Signal Weight-of-Evidence (WoE) Summary"] = ensure_single_paragraph(report.get("Signal Weight-of-Evidence (WoE) Summary", ""))

    # ----------------------------
    # Display (UI)
    # ----------------------------
    st.markdown("## Signal Validation Criteria (Table)")
    crit_df = pd.DataFrame(
        [
            ["Strength of Evidence", scores.get("Strength of Evidence", "")],
            ["Consistency", scores.get("Consistency", "")],
            ["Temporality", scores.get("Temporality", "")],
            ["Biological Plausibility", scores.get("Biological Plausibility", "")],
            ["Confounding Impact", scores.get("Confounding Impact", "")],
            ["Overall Weight of Evidence", scores.get("Overall Weight of Evidence", "")],
        ],
        columns=["Criterion", "Rating"],
    )
    st.table(crit_df)

    st.markdown("## Final Validation Outcome")
    st.write(outcome)

    st.markdown("## Report Sections")
    for k in section_keys:
        st.markdown(f"### {k}")
        st.write(str(report.get(k, "")).strip())

    # ----------------------------
    # Word Export (no evidence snapshot)
    # ----------------------------
    doc = Document()
    doc.add_heading("Signal Validation Report", level=1)
    doc.add_paragraph(f"Event PT: {pt_choice}")

    if product_name.strip() or generic_name.strip():
        doc.add_paragraph(product_context_string())

    doc.add_paragraph("")

    doc.add_heading("Signal Validation Criteria", level=2)
    t = doc.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Criterion"
    t.rows[0].cells[1].text = "Rating"
    for row in [
        ("Strength of Evidence", scores.get("Strength of Evidence", "")),
        ("Consistency", scores.get("Consistency", "")),
        ("Temporality", scores.get("Temporality", "")),
        ("Biological Plausibility", scores.get("Biological Plausibility", "")),
        ("Confounding Impact", scores.get("Confounding Impact", "")),
        ("Overall Weight of Evidence", scores.get("Overall Weight of Evidence", "")),
    ]:
        r = t.add_row().cells
        r[0].text = row[0]
        r[1].text = str(row[1])

    doc.add_paragraph("")
    doc.add_heading("Final Validation Outcome", level=2)
    doc.add_paragraph(outcome)

    doc.add_page_break()
    doc.add_heading("Narrative", level=2)
    for k in section_keys:
        doc.add_heading(k, level=3)
        doc.add_paragraph(str(report.get(k, "")).strip())

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    st.download_button(
        "Download Signal Validation Report (Word)",
        data=buf,
        file_name=f"Signal_Validation_{pt_choice}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
