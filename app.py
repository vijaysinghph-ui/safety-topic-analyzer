import os
import streamlit as st
import pandas as pd
from openai import OpenAI

st.title("Safety Topic Analyzer")

# ---------- Helper: find likely column names ----------
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


# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    # READ THE EXCEL (this is what you were missing)
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head(20))

    # ---------- Auto-detect Event PT ----------
    PT_CANDIDATES = [
        "event pt", "preferred term", "meddra pt", "reaction pt", "event_preferred_term", "pt"
    ]
    CASE_CANDIDATES = ["case number", "case id", "icsr", "report id", "safety report id"]

    pt_col = find_col(df, PT_CANDIDATES)
    case_col = find_col(df, CASE_CANDIDATES)

    st.write("Detected columns:")
    st.write({"Event PT column": pt_col, "Case column (optional)": case_col})

    if not pt_col:
        st.error("Could not detect Event PT column. Please rename your PT column to include 'Event PT' or 'Preferred Term'.")
        st.stop()

    # ---------- Group by Event PT ----------
    grouped = df.groupby(pt_col, dropna=True)
    topic_table = grouped.size().reset_index(name="rows")
    topic_table = topic_table.sort_values("rows", ascending=False)

    st.subheader("Auto-grouped topics (by Event PT)")
    st.dataframe(topic_table.head(50))

    # ---------- Optional: test OpenAI connectivity later ----------
    if st.button("Test OpenAI Connection"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": "Reply with: OK"}],
        )
        st.write(resp.choices[0].message.content)
