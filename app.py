import streamlit as st
import pandas as pd
from openai import OpenAI

st.title("Safety Topic Analyzer")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df)

    if st.button("Analyze Safety Topics"):

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""
        You are a pharmacovigilance expert.

        Analyze safety data and identify safety topics.

        Provide scientific justification suitable for PBRER.

        Data:
        {df.to_string()}
        """

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.choices[0].message.content

        st.write(result)


