import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from firebase.connect import edit_single_match
from jobs.get_all_firebase_data import load_offline_df
from datetime import datetime


load_dotenv()
AUTHENTICATION_KEY = os.environ.get('AUTHENTICATION_KEY')

current_year = datetime.now().year

def convert_to_datetime(date_str):
    return pd.to_datetime(f"{date_str}", format='%Y-%m-%d')

def get_winning_odds(x):
    if x['winner']==x['left_team_name']:
        return x['left_odds']
    if x['winner']==x['right_team_name']:
        return x['right_odds']

def get_losing_odds(x):
    if x['winner']!=x['left_team_name']:
        return x['left_odds']
    if x['winner']!=x['right_team_name']:
        return x['right_odds']

def find_matching_winning_odds(row, odds_summary):
    match = odds_summary.loc[odds_summary['winning_odds'] == row['losing_odds'], 'winning_odds']
    if not match.empty:
        return match.iloc[0]
    return None

def find_matching_winning_count(row, odds_summary):
    match = odds_summary.loc[odds_summary['winning_odds'] == row['losing_odds'], 'winning_count']
    if not match.empty:
        return match.iloc[0]
    return None

def app():
    match_odds_df = load_offline_df('match_logs')
    games = match_odds_df['game'].unique()
    selected_game = st.selectbox('Select a game to filter', games)
    is_winner_df = match_odds_df[(match_odds_df['winner'].notna()) & (match_odds_df['game']==selected_game) & ((match_odds_df['winner']!='Draw'))]
    if not is_winner_df.empty:
        is_winner_df['winning_odds'] = is_winner_df.apply(lambda x: get_winning_odds(x),axis=1)
        is_winner_df['losing_odds'] = is_winner_df.apply(lambda x: get_losing_odds(x),axis=1)
        print(is_winner_df)
        is_winner_df['winning_odds'] = is_winner_df['winning_odds'].astype(float).astype(int).astype(str)
        is_winner_df['losing_odds'] = is_winner_df['losing_odds'].astype(float).astype(int).astype(str)
        
        # Create mirror bar chart
        winning_odds_counts = is_winner_df['winning_odds'].value_counts()
        losing_odds_counts = is_winner_df['losing_odds'].value_counts()
        
        # Ensure that both counts series have the same index
        all_odds = sorted(set(winning_odds_counts.index) | set(losing_odds_counts.index))
        winning_odds_counts = winning_odds_counts.reindex(all_odds, fill_value=0)
        losing_odds_counts = losing_odds_counts.reindex(all_odds, fill_value=0)
        
        fig = go.Figure()

        # Add bars for winning odds
        fig.add_trace(go.Bar(
            y=all_odds,
            x=winning_odds_counts,
            name='Winning Odds',
            orientation='h',
            marker=dict(color='blue')
        ))

        # Add bars for losing odds (mirrored)
        fig.add_trace(go.Bar(
            y=all_odds,
            x=-losing_odds_counts,
            name='Losing Odds',
            orientation='h',
            marker=dict(color='red')
        ))

        # Update layout
        fig.update_layout(
            title='Mirror Bar Chart of Odds Counts',
            xaxis_title='Count',
            yaxis_title='Odds',
            xaxis=dict(
                title='Count',
                tickvals=[-max(losing_odds_counts), 0, max(winning_odds_counts)],
                ticktext=[f'-{max(losing_odds_counts)}', '0', f'{max(winning_odds_counts)}']
            ),
            barmode='overlay',
            bargap=0.1
        )

        # Display the chart using Streamlit
        st.plotly_chart(fig)

        odds_summary = is_winner_df.groupby(['winning_odds', 'losing_odds']).size().reset_index(name='winning_count')
        odds_summary['winning_odds_lost_to'] = odds_summary.apply(lambda x: find_matching_winning_odds(x, odds_summary), axis=1).fillna(0)
        odds_summary['winning_odds_lost_to_count'] = odds_summary.apply(lambda x: find_matching_winning_count(x, odds_summary), axis=1).fillna(0)
        odds_summary['total_matches'] = odds_summary['winning_count'] + odds_summary['winning_odds_lost_to_count']
        odds_summary['chance_to_win_%'] = ((odds_summary['winning_count'] / odds_summary['total_matches']) * 100)
        odds_summary['chance_to_win_%'] = odds_summary['chance_to_win_%'].replace([np.inf, -np.inf], np.nan).fillna(100).round().astype(int)
        odds_summary['possible_win_amount'] = ((odds_summary['winning_odds'].astype(int)+.2) * odds_summary['winning_count'].astype(int) * 85)
        odds_summary['possible_loss_amount'] = (odds_summary['winning_odds_lost_to_count'].astype(int) * -85)
        odds_summary['possible_total_amt_inv'] = (odds_summary['total_matches'] * 85)
        odds_summary['inc_or_dec_capital'] = ((odds_summary['possible_win_amount']/odds_summary['possible_total_amt_inv'])).astype(float)
        st.dataframe(odds_summary,use_container_width=True)


    ordered_columns = ['game', 'date', 'left_team_name', 'left_odds', 'right_team_name', 'right_odds', 'winner']
    match_odds_df = match_odds_df[ordered_columns]
    match_odds_df['date'] = match_odds_df['date'].apply(convert_to_datetime)

    st.title('Match Odds History')

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
                
