"""Temporary Streamlit landing page until dashboard features are implemented."""

import streamlit as st

st.set_page_config(page_title="NFL Fantasy ML Optimizer", layout="wide")
st.title("NFL Fantasy ML Optimizer")
st.info(
    "The repository scaffold is ready. Data ingestion, validated projections, "
    "lineup optimization, and waiver recommendations have not been implemented yet."
)
st.caption("No predictions are shown until the data and evaluation pipeline is implemented.")
