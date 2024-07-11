import pandas as pd
import streamlit as st
import plotly.express as px
import os
from dotenv import load_dotenv
from firebase.connect import run_odds_pipeline, edit_single_match
from datetime import datetime

load_dotenv()
AUTHENTICATION_KEY = os.environ.get('AUTHENTICATION_KEY')
current_year = datetime.now().year

def convert_to_datetime(date_str):
    return pd.to_datetime(f"{date_str} {current_year}", format='%b %d %Y')

def app():
    match_odds_df = run_odds_pipeline()
    ordered_columns = ['game', 'date', 'left_team_name', 'left_odds', 'right_team_name', 'right_odds', 'winner']
    match_odds_df = match_odds_df[ordered_columns]
    match_odds_df['date'] = match_odds_df['date'].apply(convert_to_datetime)

    st.title('Match Odds History')
    games = match_odds_df['game'].unique()
    selected_game = st.selectbox('Select a game to filter', games)

    # Bet history details table
    if selected_game:
        match_odds_df = match_odds_df[match_odds_df['game'] == selected_game]
        

    st.dataframe(match_odds_df.sort_values(by='date'))

    # Capture row selection
    selected_row = st.selectbox('Select a row to edit', match_odds_df.index)
    selected_data = match_odds_df.loc[selected_row]

    if selected_row is not None:
        with st.form("edit_form"):
            game = st.text_input("Game", selected_data['game'])
            left_team_name = st.text_input("left_team_name", value=selected_data['left_team_name'])
            left_odds = st.text_input("left_odds", value=selected_data['left_odds'])
            right_team_name = st.text_input("right_team_name", value=selected_data['right_team_name'])
            right_odds = st.text_input("right_odds", value=selected_data['right_odds'])
            winner = st.text_input("winner", value=selected_data['winner'])
            st.text_input('Please enter authentication key to be able to save the transaction', key='auth_key', type='password')
            
            updated_data = {}
            if st.form_submit_button("Save"):
                auth_key = st.session_state.get('auth_key', '')
                if auth_key != AUTHENTICATION_KEY:
                    st.error("Invalid authentication key. Transactions not saved.")
                    return
                
                updated_data['game'] = game
                updated_data['left_team_name'] = left_team_name
                updated_data['left_odds'] = left_odds
                updated_data['right_team_name'] = right_team_name
                updated_data['right_odds'] = right_odds
                updated_data['winner'] = winner

                edit_single_match(selected_row,updated_data)
                st.success("Row updated successfully!")
                
