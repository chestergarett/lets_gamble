import streamlit as st
from  frontend import mlbb_msc_stats,mlbb_predict_winner,bet_transactions,record_bets, risk_calculator, mlbb_model_ab_test

pages = {
    "MLBB Stats": mlbb_msc_stats,
    "MLBB Predict Winner": mlbb_predict_winner,
    "MLBB Prediction A/B Tests": mlbb_model_ab_test,
    "BET Record History": bet_transactions,
    "Record Bets": record_bets,
    "Risk Calculator": risk_calculator
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page.app()