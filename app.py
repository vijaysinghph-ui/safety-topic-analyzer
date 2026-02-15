import os
import io
import json
import re
import streamlit as st
import pandas as pd
from openai import OpenAI
from docx import Document

# ============================
# App Config
# ============================
st.set_page_config(page_title="Safety Topic Analyzer", layout="wide")
st.title("Safety Topic Analyzer")

SYSTEM_STYLE = """
You are acting as:
- EU QPPV-level safety physician
- Experienced in PBRER, signal validation, and regulatory inspection defense
- Writing content that may be reviewed by EMA, MHRA, or FDA

Your writing must:
- Be inspection-ready
- Be conservative and evidence-based
- Avoid speculative statements
- Avoid emotional or persuasive language
""".strip()

# ============================
# OpenAI Client
# ============================
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

# ============================
# Helpers
# ============================
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

def decision_matrix(overall_evidence: str, confounding: str):
    overall = rating_norm(overall_evidence)
    conf = rating_norm(confounding)

    if overall == "Strong":
        causality = "Supported"
        decision = "Include / Escalate for Risk Evaluation"
    elif overall == "Moderate":
        if conf == "Low":
            causality = "Possible"
            decision = "Escalate for Signal Validation"
        else:
            causality = "Possible"
            decision = "Continue Monitoring"
    elif overall == "Weak":
        if conf == "High":
            causality = "Unlikely"
            decision = "Close"
        else:
            causality = "Insufficient Evidence"
            decision = "Continue Monitoring"
    else:
        causality = "Insufficient Evidence"
        decision = "Continue Monitoring"

    return causality, decision

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

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

# ============================
# Upload Excel
# ============================
uploaded_file = st.file_uploader("Upload Excel Line Listing", type=["xlsx"])
if uploaded_file is None:
    st.info("Upload an Excel line listing to begin.")
    st.stop()

df = pd.read_excel(uploaded_file)
st.subheader("Preview of Uploaded Data")
st.dataframe(df.head(20))

# ============================
# Column detection
# ============================
PT_CANDS = ["event pt", "preferred term", "meddra pt", "reaction pt", "pt"]
SERIOUS_CANDS = ["serious case flag", "serious", "event seriousness", "seriousness"]
FATAL_CANDS = ["death flag", "fatal case flag", "fatal"]
LISTED_CANDS = ["listedness", "event listedness", "expected"]
DECH_CANDS = ["dechallenge", "dechallenge results"]
RECH_CANDS = ["rechallenge", "rechallenge results"]
ONSET_CANDS = ["onset latency", "time to onset", "event start date"]
COUNTRY_CANDS = ["country"]
NARR_CANDS = ["narrative"]
DRUG_CANDS = ["suspect product name", "suspect drug", "drug", "product", "generic name"]
CASE_CANDS = ["case number", "case id", "icsr", "report id", "safety report id"]

pt_col = find_col(df, PT_CANDS)
ser_col = find_col(df, SERIOUS_CANDS)
fat_col = find_col(df, FATAL_CANDS)
list_col = find_col(df, LISTED_CANDS)
dech_col = find_col(df, DECH_CANDS)
rech_col = find_col(df, RECH_CANDS)
onset_col = find_col(df, ONSET_CANDS)
country_col = find_col(df, COUNTRY_CANDS)
narr_col = find_col(df, NARR_CANDS)
drug_col = find_col(df, DRUG_CANDS)
case_col = find_col(df, CASE_CANDS)

if not pt_col:
    st.error("Event PT column not detected. Rename column to include 'Event PT' or 'Preferred Term'.")
    st.stop()

topic_table = (
    df.groupby(pt_col, dropna=True)
    .size()
    .reset_index(name="row_count")
    .sort_values("row_count", ascending=False)
)
st.subheader("Auto-grouped Topics (by Event PT)")
st.dataframe(topic_table.head(50))

# ============================
# Sidebar: Product context
# ============================
st.sidebar.header("Product Context (stabilizes Background)")
product_name = st.sidebar.text_input("Suspect Product (brand or substance)", value="")
generic_name = st.sidebar.text_input("Generic name (optional)", value="")
ther_class = st.sidebar.text_input("Therapeutic class (optional)", value="")

def product_context_string():
    parts = []
    if product_name.strip():
        parts.append(f"Suspect product: {product_name.strip()}")
    if generic_name.strip():
        parts.append(f"Generic name: {generic_name.strip()}")
    if ther_class.strip():
        parts.append(f"Therapeutic class: {ther_class.strip()}")
    if not parts:
        parts.append("Suspect product: Not provided (keep background high-level and cautious).")
    return "\n".join(parts)

# ============================
# Evidence builder
# ============================
def build_evidence(subset, pt_value):
    evidence = {
        "event_pt": str(pt_value),
        "row_count": int(len(subset)),
        "unique_case_count": int(subset[case_col].nunique()) if case_col else None,
        "serious_yes": yes_count(subset[ser_col]) if ser_col else None,
        "fatal_yes": yes_count(subset[fat_col]) if fat_col else None,
        "listedness_top": top_values(subset[list_col]) if list_col else None,
        "dechallenge_top": top_values(subset[dech_col]) if dech_col else None,
        "rechallenge_top": top_values(subset[rech_col]) if rech_col else None,
        "onset_examples": subset[onset_col].dropna().astype(str).head(5).tolist() if onset_col else [],
        "countries_top": top_values(subset[country_col]) if country_col else None,
        "narrative_snippets": subset[narr_col].dropna().astype(str).head(3).tolist() if narr_col else [],
    }
    if drug_col:
        evidence["suspect_products_top"] = top_values(subset[drug_col])
    return evidence

# ============================
# Modes
# ============================
st.subheader("Choose Output Mode")
mode = st.radio(
    "Mode",
    [
        "Signal Assessment (Structured + Scoring)",
        "PBRER Summary (Aggregate Synopsis)",
        "Bulk Trending (Monthly Scan)",
    ],
)

# ==================================================
# MODE 1: Signal Assessment (Structured + Scoring)
# ==================================================
if mode.startswith("Signal Assessment"):
    st.subheader("Signal Assessment (Structured + Scoring)")

    pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

    st.markdown("#### Optional: lock standardized outputs (recommended)")
    lock_causality = st.selectbox(
        "Causality Conclusion (optional)",
        ["(Use matrix result)", "Supported", "Possible", "Insufficient Evidence", "Unlikely"],
        index=0,
    )
    lock_decision = st.selectbox(
        "Safety Topic Decision (optional)",
        ["(Use matrix result)", "Include", "Continue Monitoring", "Close", "Escalate for Signal Validation", "Include / Escalate for Risk Evaluation"],
        index=0,
    )

    if st.button("Generate Signal Assessment + Word"):
        subset = df[df[pt_col] == pt_choice]
        evidence = build_evidence(subset, pt_choice)

        # ---- Step 1: Scoring JSON ----
        score_keys = [
            "Strength",
            "Consistency",
            "Temporality",
            "Plausibility",
            "Confounding",
            "Overall Evidence",
            "Rationale (brief)"
        ]

        score_prompt = f"""
Return ONLY valid JSON with EXACT keys:

{{
  "Strength": "Weak/Moderate/Strong",
  "Consistency": "Weak/Moderate/Strong",
  "Temporality": "Weak/Moderate/Strong/Not assessable",
  "Plausibility": "Weak/Moderate/Strong",
  "Confounding": "Low/Moderate/High",
  "Overall Evidence": "Weak/Moderate/Strong",
  "Rationale (brief)": "2-4 sentences, evidence-based, inspection-ready"
}}

Use only provided evidence. No speculation.

EVIDENCE:
{evidence}
"""
        raw_scores = call_openai(score_prompt)
        scores = json_or_none(raw_scores)
        if scores is None or not must_have_keys(scores, score_keys):
            repair = call_openai("Convert into ONLY valid JSON with the exact keys. Output JSON only.\n\n" + raw_scores)
            scores = json_or_none(repair)

        if scores is None or not must_have_keys(scores, score_keys):
            st.error("Could not parse scoring JSON. Raw output below:")
            st.code(raw_scores)
            st.stop()

        # deterministic outputs
        matrix_causality, matrix_decision = decision_matrix(scores.get("Overall Evidence", ""), scores.get("Confounding", ""))

        # allow locks
        final_causality = matrix_causality if lock_causality == "(Use matrix result)" else lock_causality
        final_decision = matrix_decision if lock_decision == "(Use matrix result)" else lock_decision

        # ---- Step 2: Structured Signal Assessment (JSON) ----
        section_keys = [
            "Background of Drug-Event Combination",
            "Case Synopsis",
            "Bradford Hill Assessment",
            "Regulatory Implications",
            "Causality Conclusion",
            "Safety Topic Decision",
            "Recommended Next Action",
        ]

        structured_prompt = f"""
Return ONLY valid JSON (no markdown) with EXACT keys:

{{
  "Background of Drug-Event Combination": "string",
  "Case Synopsis": "string",
  "Bradford Hill Assessment": "string",
  "Regulatory Implications": "string",
  "Causality Conclusion": "string",
  "Safety Topic Decision": "string",
  "Recommended Next Action": "string"
}}

RULES:
- Output ALL keys in EXACT order above.
- Background: product-level only, <=120 words, single paragraph, no case counts.
- Regulatory Implications: NO browsing; do not invent safety communications.
  If uncertain, write exactly:
  "No widely recognized regulatory action specific to this drug-event combination."
- Use EXACT standardized outputs:
  Causality Conclusion: "{final_causality}"
  Safety Topic Decision: "{final_decision}"
- Do not output bullet-only sections; short paragraphs are preferred.

PRODUCT CONTEXT:
{product_context_string()}

SCORING (for context only; do not restate as a table):
{scores}

EVIDENCE:
{evidence}
"""
        raw_struct = call_openai(structured_prompt)
        data = json_or_none(raw_struct)
        if data is None or not must_have_keys(data, section_keys):
            repair = call_openai("Convert into ONLY valid JSON with the exact keys. Output JSON only.\n\n" + raw_struct)
            data = json_or_none(repair)

        if data is None or not must_have_keys(data, section_keys):
            st.error("Could not parse structured JSON. Raw output below:")
            st.code(raw_struct)
            st.stop()

        # enforce background formatting
        data["Background of Drug-Event Combination"] = ensure_single_paragraph(data.get("Background of Drug-Event Combination", ""))

        # Display
        st.markdown("### Scoring + Matrix")
        st.write({"Scoring": scores, "Matrix Causality": matrix_causality, "Matrix Decision": matrix_decision})
        st.markdown("### Final Standardized Outputs")
        st.write({"Causality Conclusion": final_causality, "Safety Topic Decision": final_decision})

        st.subheader(f"Signal Assessment – {pt_choice}")
        for k in section_keys:
            st.markdown(f"### {k}")
            st.write(str(data.get(k, "")).strip())

        # Word export
        doc = Document()
        doc.add_heading("Signal Assessment Report", level=1)
        doc.add_paragraph(f"Event PT: {pt_choice}")
        doc.add_paragraph("")

        doc.add_heading("Scoring + Matrix", level=2)
        for k in score_keys:
            doc.add_paragraph(f"{k}: {scores.get(k, '')}")
        doc.add_paragraph(f"Matrix Causality: {matrix_causality}")
        doc.add_paragraph(f"Matrix Decision: {matrix_decision}")
        doc.add_paragraph(f"Final Causality Conclusion: {final_causality}")
        doc.add_paragraph(f"Final Safety Topic Decision: {final_decision}")

        doc.add_page_break()
        doc.add_heading("Structured Assessment", level=2)
        for k in section_keys:
            doc.add_heading(k, level=3)
            doc.add_paragraph(str(data.get(k, "")).strip())

        doc.add_page_break()
        doc.add_heading("Evidence Snapshot", level=2)
        doc.add_paragraph(json.dumps(evidence, indent=2))

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.download_button(
            "Download Signal Assessment (Word)",
            data=buf,
            file_name=f"Signal_Assessment_{pt_choice}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

# ==================================================
# MODE 2: PBRER Summary (Aggregate Synopsis)
# ==================================================
elif mode.startswith("PBRER Summary"):
    st.subheader("PBRER Summary (Aggregate Synopsis)")

    pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

    if st.button("Generate PBRER Summary + Word"):
        subset = df[df[pt_col] == pt_choice]
        evidence = build_evidence(subset, pt_choice)

        prompt = f"""
Write a PBRER-ready aggregate synopsis for the reporting interval.

Return ONLY valid JSON with EXACT keys:
{{
  "PBRER Summary": "string"
}}

RULES for "PBRER Summary":
- ONE continuous paragraph (no bullets, no line breaks, no labels, no colons like 'Decision:')
- 140–190 words
- Include: case counts (use unique_case_count if provided, otherwise row_count), seriousness, fatalities, geography, listedness, key clinical pattern(s), confounders/alternative etiologies, and an overall benefit–risk position sentence at the end.
- Do NOT mention Bradford Hill explicitly.
- Do NOT invent regulatory actions.
- Use only evidence provided.

PRODUCT CONTEXT:
{product_context_string()}

EVIDENCE:
{evidence}
"""
        raw = call_openai(prompt)
        data = json_or_none(raw)
        if data is None or "PBRER Summary" not in data:
            repair = call_openai("Convert into ONLY valid JSON with the exact key. Output JSON only.\n\n" + raw)
            data = json_or_none(repair)

        if data is None or "PBRER Summary" not in data:
            st.error("Could not parse PBRER Summary JSON. Raw output below:")
            st.code(raw)
            st.stop()

        summary = ensure_single_paragraph(data["PBRER Summary"])
        wc = word_count(summary)
        if wc < 130 or wc > 210:
            fix = call_openai(
                f"Rewrite into ONE paragraph, 140–190 words, no bullets/labels/line breaks, no new facts. End with benefit–risk sentence.\n\nTEXT:\n{summary}"
            )
            summary = ensure_single_paragraph(fix)

        st.subheader(f"PBRER Summary – {pt_choice}")
        st.write(summary)

        doc = Document()
        doc.add_heading("PBRER Summary (Aggregate Synopsis)", level=1)
        doc.add_paragraph(f"Event PT: {pt_choice}")
        doc.add_paragraph("")
        doc.add_paragraph(summary)

        doc.add_page_break()
        doc.add_heading("Evidence Snapshot", level=2)
        doc.add_paragraph(json.dumps(evidence, indent=2))

        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.download_button(
            "Download PBRER Summary (Word)",
            data=buf,
            file_name=f"PBRER_Summary_{pt_choice}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

# ==================================================
# MODE 3: Bulk Trending (unchanged)
# ==================================================
else:
    st.subheader("Bulk Trending (Monthly Scan)")
    TOP_N = st.slider("Number of PTs to Analyze", 3, 30, 10)

    if st.button("Analyze Top PTs"):
        top_pts = topic_table.head(TOP_N)[pt_col].tolist()

        for pt in top_pts:
            subset = df[df[pt_col] == pt]
            evidence = build_evidence(subset, pt)

            prompt = f"""
Provide:
1) Event PT
2) Safety topic decision (Include / Continue Monitoring / Close / Escalate)
3) Key evidence (max 4 bullets)
4) Causality conclusion (Supported / Possible / Insufficient Evidence / Unlikely)
5) Recommended action

Evidence:
{evidence}
"""
            out = call_openai(prompt)
            st.subheader(f"Trending – {pt}")
            st.write(out)
            st.markdown("---")
