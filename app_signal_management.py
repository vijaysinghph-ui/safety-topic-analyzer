import streamlit as st
import pandas as pd

st.set_page_config(page_title="Signal Management Tool 2", layout="wide")
st.title("Signal Management Tool 2 (Prototype)")

uploaded = st.file_uploader("Upload Excel Line Listing", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    st.success(f"Loaded {len(df)} rows.")
    st.dataframe(df.head(50))
