import streamlit as st
from  frontend import mlbb_msc_stats,mlbb_predict_winner


pages = {
    "MLBB Stats": mlbb_msc_stats,
    "MLBB Predict Winner": mlbb_predict_winner
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page.app()