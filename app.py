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

# ============================
# Hidden governance layer (NOT shown in UI)
# ============================
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

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

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

def extract_scores(data: dict):
    keys = ["Strength", "Consistency", "Temporality", "Plausibility", "Confounding", "Overall Evidence"]
    out = {}
    for k in keys:
        out[k] = rating_norm(data.get(k, ""))
    return out

def must_have_keys(d, keys):
    return isinstance(d, dict) and all(k in d for k in keys)

def call_openai(user_prompt: str, model: str = "gpt-5") -> str:
    """Single place to enforce SYSTEM_STYLE everywhere."""
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_STYLE},
            {"role": "user", "content": user_prompt},
        ],
    )
    return r.choices[0].message.content

# ============================
# UI: Upload
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
PT_CANDIDATES = ["event pt", "preferred term", "meddra pt", "reaction pt", "pt"]
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

pt_col = find_col(df, PT_CANDIDATES)
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

# ============================
# Topic table
# ============================
topic_table = (
    df.groupby(pt_col, dropna=True)
    .size()
    .reset_index(name="row_count")
    .sort_values("row_count", ascending=False)
)

st.subheader("Auto-grouped Topics (by Event PT)")
st.dataframe(topic_table.head(50))

# ============================
# Mode selection
# ============================
st.subheader("Choose Output Mode")
mode = st.radio(
    "Mode",
    [
        "Signal Assessment (Scored + Deterministic Decision)",
        "PBRER Narrative (Document Writer)",
        "Bulk Trending (Monthly Scan)",
    ],
)

# ============================
# Evidence builder
# ============================
def build_evidence(subset, pt_value):
    evidence = {
        "event_pt": str(pt_value),
        "row_count": int(len(subset)),  # rows, not necessarily unique cases
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
# Shared UI inputs for product context
# ============================
st.sidebar.header("Product Context (helps stabilize Background)")
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

# ==================================================
# MODE 1: Signal Assessment (scored) + deterministic decision
# ==================================================
if mode.startswith("Signal"):
    st.subheader("Signal Assessment (Scoring + Deterministic Conclusion)")

    pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

    if st.button("Run Signal Assessment"):
        subset = df[df[pt_col] == pt_choice]
        evidence = build_evidence(subset, pt_choice)

        score_schema_keys = [
            "Strength",
            "Consistency",
            "Temporality",
            "Plausibility",
            "Confounding",
            "Overall Evidence",
            "Rationale (brief)"
        ]

        score_prompt = f"""
Return ONLY valid JSON (no markdown, no extra text) with EXACT keys:

{{
  "Strength": "Weak/Moderate/Strong",
  "Consistency": "Weak/Moderate/Strong",
  "Temporality": "Weak/Moderate/Strong/Not assessable",
  "Plausibility": "Weak/Moderate/Strong",
  "Confounding": "Low/Moderate/High",
  "Overall Evidence": "Weak/Moderate/Strong",
  "Rationale (brief)": "2-4 sentences, evidence-based. No speculation."
}}

Rules:
- Use only the evidence provided. If not assessable, say "Not assessable".
- Keep rationale short and inspection-ready.

EVIDENCE:
{evidence}
"""
        raw = call_openai(score_prompt)
        scores = json_or_none(raw)

        if scores is None or not must_have_keys(scores, score_schema_keys):
            repair_raw = call_openai(
                "Convert into ONLY valid JSON with the exact keys and allowed values. Output JSON only.\n\n" + raw
            )
            scores = json_or_none(repair_raw)

        if scores is None or not must_have_keys(scores, score_schema_keys):
            st.error("Could not parse scoring JSON. Raw output below:")
            st.code(raw)
            st.stop()

        norm_scores = extract_scores(scores)
        causality, decision = decision_matrix(norm_scores["Overall Evidence"], norm_scores["Confounding"])

        st.markdown("### Scoring Output")
        st.write(norm_scores)
        st.markdown("### Deterministic Outputs (Matrix)")
        st.write({"Causality Conclusion": causality, "Safety Topic Decision": decision})
        st.markdown("### Rationale (brief)")
        st.write(scores.get("Rationale (brief)", ""))

        # Word export
        document = Document()
        document.add_heading("Signal Assessment Report", level=1)
        document.add_paragraph(f"Event PT: {pt_choice}")
        document.add_paragraph("")

        document.add_heading("Scoring", level=2)
        for k in ["Strength", "Consistency", "Temporality", "Plausibility", "Confounding", "Overall Evidence"]:
            document.add_paragraph(f"{k}: {norm_scores.get(k, '')}")

        document.add_heading("Deterministic Conclusion (Policy Matrix)", level=2)
        document.add_paragraph(f"Causality Conclusion: {causality}")
        document.add_paragraph(f"Safety Topic Decision: {decision}")

        document.add_heading("Rationale (brief)", level=2)
        document.add_paragraph(str(scores.get("Rationale (brief)", "")).strip())

        document.add_page_break()
        document.add_heading("Evidence Snapshot", level=2)
        document.add_paragraph(json.dumps(evidence, indent=2))

        buf = io.BytesIO()
        document.save(buf)
        buf.seek(0)

        st.download_button(
            "Download Signal Assessment (Word)",
            data=buf,
            file_name=f"Signal_Assessment_{pt_choice}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

# ==================================================
# MODE 2: PBRER Narrative (writer)
# ==================================================
elif mode.startswith("PBRER"):
    st.subheader("PBRER Narrative (Regulatory Writer)")

    pt_choice = st.selectbox("Select Event PT", topic_table[pt_col].tolist())

    st.markdown("#### Optional: lock conclusions (recommended for consistency)")
    locked_causality = st.selectbox(
        "Causality Conclusion (optional)",
        ["(Let AI infer)", "Supported", "Possible", "Insufficient Evidence", "Unlikely"],
        index=0,
    )
    locked_decision = st.selectbox(
        "Safety Topic Decision (optional)",
        ["(Let AI infer)", "Include", "Continue Monitoring", "Close", "Escalate for Signal Validation", "Include / Escalate for Risk Evaluation"],
        index=0,
    )

    if st.button("Generate PBRER Narrative + Word"):
        subset = df[df[pt_col] == pt_choice]
        evidence = build_evidence(subset, pt_choice)

        schema_keys = [
            "Background of Drug-Event Combination",
            "Case Synopsis",
            "Bradford Hill Assessment",
            "Regulatory Landscape Overview",
            "Causality Conclusion",
            "Safety Topic Decision",
            "PBRER-Ready Summary",
            "Recommended Next Action",
        ]

        lock_block = ""
        if locked_causality != "(Let AI infer)":
            lock_block += f"\nLocked Causality Conclusion: {locked_causality}"
        if locked_decision != "(Let AI infer)":
            lock_block += f"\nLocked Safety Topic Decision: {locked_decision}"

        narrative_prompt = f"""
Return ONLY valid JSON (no markdown, no extra text) with EXACT keys:

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
1) Background of Drug-Event Combination MUST be product-level ONLY and ≤120 words:
   - pharmacologic class/mechanism relevant to the event
   - known class effects
   - general labeling recognition if broadly known
   - NO case counts, countries, listedness, seriousness, timing, or reporting interval details.

2) Case Synopsis MUST contain ALL reporting interval evidence:
   - totals, seriousness, geography, listedness (if available), patterns, confounders.
   - No pharmacology here.

3) Regulatory Landscape Overview:
   - Do NOT fabricate regulatory actions.
   - If uncertain, write exactly:
     "No widely recognized regulatory action specific to this drug-event combination."

4) If Locked Causality/Decision are provided, use them EXACTLY and do not change wording.

5) PBRER-Ready Summary:
   - ONE continuous paragraph (NO bullets, NO line breaks, NO labels like "Decision:")
   - 150–180 words
   - Integrate evidence + Hill reasoning + causality + decision
   - Do NOT introduce new facts
   - End with explicit benefit–risk position statement

PRODUCT CONTEXT:
{product_context_string()}

{lock_block}

REPORTING INTERVAL EVIDENCE:
{evidence}
"""
        raw = call_openai(narrative_prompt)
        data = json_or_none(raw)

        if data is None or not must_have_keys(data, schema_keys):
            repair_raw = call_openai(
                "Convert into ONLY valid JSON with the exact keys. Output JSON only.\n\n" + raw
            )
            data = json_or_none(repair_raw)

        if data is None or not must_have_keys(data, schema_keys):
            st.error("Could not parse narrative JSON. Raw output below:")
            st.code(raw)
            st.stop()

        # enforce summary paragraph + length
        summary = ensure_paragraph(data.get("PBRER-Ready Summary", ""))
        wc = word_count(summary)

        if wc < 140 or wc > 200:
            fix_prompt = f"""
Rewrite into ONE continuous regulatory paragraph, 150–180 words.
No bullets, no headings, no line breaks, no labels.
Do not add new facts. End with benefit–risk position.

TEXT:
{summary}
"""
            fixed = call_openai(fix_prompt)
            summary = ensure_paragraph(fixed)

        data["PBRER-Ready Summary"] = summary

        # display
        st.subheader(f"PBRER Narrative – {pt_choice}")
        for k in schema_keys:
            st.markdown(f"### {k}")
            st.write(str(data.get(k, "")).strip())

        # word export
        document = Document()
        document.add_heading("PBRER Safety Topic Narrative", level=1)
        document.add_paragraph(f"Event PT: {pt_choice}")
        document.add_paragraph("")

        for k in schema_keys:
            document.add_heading(k, level=2)
            val = str(data.get(k, "")).strip()
            if k == "PBRER-Ready Summary":
                document.add_paragraph(ensure_paragraph(val))
            else:
                parts = [p for p in val.split("\n") if p.strip()]
                if not parts:
                    document.add_paragraph("")
                else:
                    for p in parts:
                        document.add_paragraph(p)

        document.add_page_break()
        document.add_heading("Evidence Snapshot", level=2)
        document.add_paragraph(json.dumps(evidence, indent=2))

        buf = io.BytesIO()
        document.save(buf)
        buf.seek(0)

        st.download_button(
            "Download PBRER Narrative (Word)",
            data=buf,
            file_name=f"PBRER_Narrative_{pt_choice}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

# ==================================================
# MODE 3: Bulk Trending
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
