import streamlit as st
from  frontend import mlbb_msc_stats,mlbb_predict_winner,bet_transactions,record_bets, risk_calculator, mlbb_model_ab_test,match_odds_logs

pages = {
    "BET Record History": bet_transactions,
    "Match Odds History": match_odds_logs,
    "Risk Calculator": risk_calculator,
    "Record Bets": record_bets,
    "MLBB Prediction A/B Tests": mlbb_model_ab_test,
    "MLBB Predict Winner": mlbb_predict_winner,
    "MLBB Stats": mlbb_msc_stats,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page.app()