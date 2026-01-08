import streamlit as st

st.set_page_config(page_title="E-Commerce Dashboard (Olist)", layout="wide")

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1400px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page("pages/0_About_Project.py", title="About Project"),
    st.Page("pages/1_Overview.py", title="Overview"),
    st.Page("pages/2_Sentiment.py", title="Sentiment"),
    st.Page("pages/3_Forecast.py", title="Forecast"),
]

pg = st.navigation(pages)
pg.run()