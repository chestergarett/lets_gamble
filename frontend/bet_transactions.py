import pandas as pd
import streamlit as st
import plotly.express as px
import os
from dotenv import load_dotenv
from firebase.connect import edit_single_bet
from jobs.get_all_firebase_data import load_offline_df

load_dotenv()
AUTHENTICATION_KEY = os.environ.get('AUTHENTICATION_KEY')

def app():
    bet_df = load_offline_df('bet_logs')

    ordered_columns = ['game', 'tournament', 'bet_amount', 'odds', 'win_loss_code', 'win_loss_amount', 'bet_with', 'bet_against']
    bet_df = bet_df[ordered_columns]
    bet_df['win_loss_amount'] = bet_df['win_loss_amount'].astype(float)
    bet_df['bet_amount'] = bet_df['bet_amount'].astype(float)
    bet_df = bet_df.sort_values(by='win_loss_code', ascending=False)

    st.title('BET History')
    # Dropdown to filter by game
    games = bet_df['game'].unique()
    selected_game = st.selectbox('Select a game to filter', ['All'] + list(games))

    if selected_game != 'All':
        bet_df = bet_df[bet_df['game'] == selected_game]
        
    # win/loss pie chart
    col1, col2 = st.columns(2)
    with col1:
        filtered_bet_df = bet_df[bet_df['win_loss_code'].isin(['WIN', 'LOSS'])]
        win_loss_counts = filtered_bet_df['win_loss_code'].value_counts().reset_index()
        win_loss_counts.columns = ['win_loss_code', 'count']
        fig = px.pie(win_loss_counts, names='win_loss_code', values='count', title='Win/Loss Distribution', color='win_loss_code',
                    color_discrete_map={'WIN': 'green', 'LOSS': 'red'})
        st.plotly_chart(fig)

    with col2:
        #insert bar chart for win_amount and loss_amount
        filtered_amounts_df = bet_df[bet_df['win_loss_code'].isin(['WIN', 'LOSS'])]
        win_loss_amounts = filtered_amounts_df.groupby('win_loss_code')['win_loss_amount'].sum().reset_index()
        # Create bar chart for win_loss_amount
        bar_fig = px.bar(win_loss_amounts, x='win_loss_code', y='win_loss_amount', title='Total Win/Loss Amount',
                         color='win_loss_code', color_discrete_map={'WIN': 'green', 'LOSS': 'red'})

        st.plotly_chart(bar_fig)
    
    total_wins = bet_df[bet_df['win_loss_code'] == 'WIN']['win_loss_amount'].sum()
    total_bet_amounts = bet_df[bet_df['win_loss_code'].isin(['WIN', 'LOSS'])]['bet_amount'].sum()
    percentage_gain = (total_wins) / total_bet_amounts - 1

    if percentage_gain >= 0:
        st.markdown(f"<div style='font-weight: bold'>% Gain: </div> <div style='text-align: center; color:green; font-size: 24px'>{percentage_gain:.2%}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-weight: bold'>% Loss: </div> <div style='text-align: center; color:red; font-size: 24px'>{percentage_gain:.2%}</div>", unsafe_allow_html=True)

    # Bet history details table
    st.dataframe(bet_df)

    # Capture row selection
    selected_row = st.selectbox('Select a row to edit', bet_df.index)
    selected_data = bet_df.loc[selected_row]

    if selected_row is not None:
        with st.form("edit_form"):
            game = st.text_input("Game", selected_data['game'])
            tournament = st.text_input("Tournament", selected_data['tournament'])
            bet_amount = st.number_input("Bet Amount", value=selected_data['bet_amount'])
            odds = st.number_input("Odds", value=selected_data['odds'])
            win_loss_options = ['WIN', 'LOSS', None]
            default_index = win_loss_options.index(selected_data['win_loss_code']) if selected_data['win_loss_code'] in win_loss_options else None
            win_loss_code = st.selectbox("Win/Loss Code", options=win_loss_options, index=default_index)
            win_loss_amount = st.number_input("Win/Loss Amount", value=selected_data['win_loss_amount'])
            bet_with = st.text_input("Bet With", selected_data['bet_with'])
            bet_against = st.text_input("Bet Against", selected_data['bet_against'])
            st.text_input('Please enter authentication key to be able to save the transaction', key='auth_key', type='password')
            
            updated_data = {}
            if st.form_submit_button("Save"):
                auth_key = st.session_state.get('auth_key', '')
                if auth_key != AUTHENTICATION_KEY:
                    st.error("Invalid authentication key. Transactions not saved.")
                    return
                
                updated_data['game'] = game
                updated_data['tournament'] = tournament
                updated_data['bet_amount'] = bet_amount
                updated_data['odds'] = odds
                updated_data['win_loss_code'] = win_loss_code
                updated_data['win_loss_amount'] = win_loss_amount
                updated_data['bet_with'] = bet_with
                updated_data['bet_against'] = bet_against

                edit_single_bet(selected_row,updated_data)
                st.success("Row updated successfully!")
                
