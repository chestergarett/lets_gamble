import streamlit as st
from  frontend import mlbb_msc_stats,mlbb_predict_winner,bet_transactions,record_bets

pages = {
    "MLBB Stats": mlbb_msc_stats,
    "MLBB Predict Winner": mlbb_predict_winner,
    "BET Record History": bet_transactions,
    "Record Bets": record_bets
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page.app()