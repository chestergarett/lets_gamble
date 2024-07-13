import pandas as pd
import streamlit as st
import plotly.express as px
import os
from dotenv import load_dotenv
from jobs.get_all_firebase_data import load_offline_df
from firebase.connect import edit_single_prediction

load_dotenv()
AUTHENTICATION_KEY = os.environ.get('AUTHENTICATION_KEY')

def app():
    pred_df = load_offline_df('model_prediction_logs')

    ordered_columns = ['model', 'team1', 'team2', 'predicted_winner', 'actual_winner', 'tournament']
    pred_df = pred_df[ordered_columns]

    st.dataframe(pred_df)

    # Capture row selection
    selected_row = st.selectbox('Select a row to edit', pred_df.index)
    selected_data = pred_df.loc[selected_row]

    if selected_row is not None:
        with st.form("edit_form"):
            model = st.text_input("model", selected_data['model'])
            team1 = st.text_input("team1", selected_data['team1'])
            team2 = st.text_input("team2", selected_data['team2'])
            tournament = st.text_input("tournament", selected_data['tournament'])
            predicted_winner = st.text_input("predicted_winner", selected_data['predicted_winner'])
            actual_winner = st.text_input("actual_winner", selected_data['actual_winner'])
            st.text_input('Please enter authentication key to be able to save the transaction', key='auth_key', type='password')
            
            updated_data = {}
            if st.form_submit_button("Save"):
                auth_key = st.session_state.get('auth_key', '')
                if auth_key != AUTHENTICATION_KEY:
                    st.error("Invalid authentication key. Transactions not saved.")
                    return
                
                updated_data['model'] = model
                updated_data['team1'] = team1
                updated_data['team2'] = team2
                updated_data['tournament'] = tournament
                updated_data['predicted_winner'] = predicted_winner
                updated_data['actual_winner'] = actual_winner

                edit_single_prediction(selected_row,updated_data)
                st.success("Row updated successfully!")
                
