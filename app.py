import streamlit as st
from  frontend import mlbb_msc_stats,mlbb_predict_winner,bet_transactions,record_bets, risk_calculator, mlbb_model_ab_test,match_odds_logs,home,mlbb_predict_winner_mpl

pages = {
    "Home": home,
    "BET Record History": bet_transactions,
    "Match Odds History": match_odds_logs,
    "Risk Calculator": risk_calculator,
    "Record Bets": record_bets,
    "MLBB International A/B Tests": mlbb_model_ab_test,
    "MLBB International Predict Winner": mlbb_predict_winner,
    "MLBB International Stats": mlbb_msc_stats,
    "MLBB Local Predict Winner": mlbb_predict_winner_mpl
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page.app()