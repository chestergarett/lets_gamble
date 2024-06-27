import pandas as pd
import streamlit as st
import plotly.express as px
from firebase.connect import run_firebase_pipeline

def app():
    bet_df = run_firebase_pipeline()
    ordered_columns = ['game', 'tournament', 'bet_amount', 'odds', 'win_loss_code', 'win_loss_amount', 'bet_with', 'bet_against']
    bet_df = bet_df[ordered_columns]
    st.title('BET History')
    st.dataframe(bet_df)
    return