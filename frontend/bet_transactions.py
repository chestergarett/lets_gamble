import pandas as pd
import streamlit as st
import plotly.express as px
import os
from dotenv import load_dotenv
from firebase.connect import edit_single_bet
from jobs.get_all_firebase_data import load_offline_df
from itertools import combinations
load_dotenv()
AUTHENTICATION_KEY = os.environ.get('AUTHENTICATION_KEY')

def simulate_bet(odds_list, df):
    filtered_df = df[df['odds_str'].isin(odds_list)]
    total_win_loss_amount = filtered_df['win_loss_amount'].sum()
    total_capital_expended = filtered_df['bet_amount'].sum()
    return total_win_loss_amount,total_capital_expended

def calculate_net_gain_loss(row):
    if row['win_loss_code'] == 'WIN':
        return row['win_loss_amount'] - row['bet_amount']
    elif row['win_loss_code'] == 'LOSS':
        return row['win_loss_amount']
    else:
        return row['win_loss_amount'] - row['bet_amount']
    
def app():
    bet_df = load_offline_df('bet_logs')
    ordered_columns = ['game', 'tournament', 'bet_amount', 'odds', 'win_loss_code', 'win_loss_amount', 'bet_with', 'bet_against', 'bet_date']
    bet_df = bet_df[ordered_columns]
    bet_df['win_loss_amount'] = bet_df['win_loss_amount'].astype(float)
    bet_df['bet_amount'] = bet_df['bet_amount'].astype(float)
    bet_df['bet_date'] = pd.to_datetime(bet_df['bet_date'])
    bet_df = bet_df.sort_values(by='win_loss_code', ascending=False)

    st.title('Betting Performance')

    # Dropdown to filter by game
    games = bet_df['game'].unique()
    selected_game = st.selectbox('Select a game to filter', ['All'] + list(games))

    # Filter tournaments based on selected game
    if selected_game != 'All':
        filtered_tournaments = bet_df[bet_df['game'] == selected_game]['tournament'].unique()
    else:
        filtered_tournaments = bet_df['tournament'].unique()
    
    selected_tournament = st.selectbox('Select a tournament to filter', ['All'] + list(filtered_tournaments))

    # Apply filters
    if selected_game != 'All':
        bet_df = bet_df[bet_df['game'] == selected_game]
    if selected_tournament != 'All':
        bet_df = bet_df[bet_df['tournament'] == selected_tournament]
        
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
        # Insert bar chart for win_amount and loss_amount
        filtered_amounts_df = bet_df[bet_df['win_loss_code'].isin(['WIN', 'LOSS'])]
        win_loss_amounts = filtered_amounts_df.groupby('win_loss_code')['win_loss_amount'].sum().reset_index()
        bar_fig = px.bar(win_loss_amounts, x='win_loss_code', y='win_loss_amount', title='Total Win/Loss Amount',
                         color='win_loss_code', color_discrete_map={'WIN': 'green', 'LOSS': 'red'})

        st.plotly_chart(bar_fig)
    
    bet_df['net_gain_loss'] = bet_df.apply(calculate_net_gain_loss, axis=1)
    total_wins = bet_df[bet_df['win_loss_code'] == 'WIN']['win_loss_amount'].sum()
    total_bet_amounts = bet_df[bet_df['win_loss_code'].isin(['WIN', 'LOSS'])]['bet_amount'].sum()
    percentage_gain = (total_wins-total_bet_amounts) / total_bet_amounts

    if percentage_gain >= 0:
        st.markdown(f"<div style='font-weight: bold'>% Gain: </div> <div style='text-align: center; color:green; font-size: 24px'>{percentage_gain:.2%}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight: bold'>Total Wins: </div> <div style='text-align: center; color:green; font-size: 24px'>{round(total_wins,2)}</div><div style='font-weight: bold'>Total Bets: </div> <div style='text-align: center; color:green; font-size: 24px'>{total_bet_amounts}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-weight: bold'>% Loss: </div> <div style='text-align: center; color:red; font-size: 24px'>{percentage_gain:.2%}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-weight: bold'>Total Wins: </div> <div style='text-align: center; color:green; font-size: 24px'>{round(total_wins,2)}</div><div style='font-weight: bold'>Total Bets: </div> <div style='text-align: center; color:green; font-size: 24px'>{total_bet_amounts}</div>", unsafe_allow_html=True)

    # Analysis
    st.title('Post Bet Analysis')
    bet_df['odds_str'] = bet_df['odds'].astype(int).astype(str)
    odds_win_rate_df = bet_df.groupby(['odds_str', 'win_loss_code'])[['win_loss_amount', 'bet_amount']].sum().reset_index()
    odds_win_rate_df['bet_amount'] = odds_win_rate_df.apply(
        lambda row: 
            0 if row['win_loss_code'] == 'WIN' else row['bet_amount'],
            axis=1
    )
    
    fig = px.bar(
        odds_win_rate_df,
        x='odds_str',
        y='win_loss_amount',
        color='win_loss_code',
        barmode='group',
        labels={'odds_str': 'Odds', 'win_loss_amount': 'Win/Loss Amount', 'win_loss_code': 'Win/Loss Code'},
        title='Win/Loss Amount by Odds'
    )

    st.plotly_chart(fig)

    # Line chart for win_loss_amount over time
    st.title('Win/Loss Amount Over Time')
    grouped_df = bet_df.groupby(['game', 'tournament', 'bet_date'])['win_loss_amount'].sum().reset_index()
    line_fig = px.line(grouped_df, x='bet_date', y='win_loss_amount', color='tournament', title='Win/Loss Amount Over Time per Tournament', markers=True)
    st.plotly_chart(line_fig)

    # Simulation
    unique_odds = odds_win_rate_df['odds_str'].unique()
    all_combinations = []
    for r in range(1, len(unique_odds) + 1):
        combinations_list = list(combinations(unique_odds, r))
        all_combinations.extend(combinations_list)

    simulation_results = []
    for combination in all_combinations:
        total_win_loss_amount, total_capital_expended = simulate_bet(combination, odds_win_rate_df)
        simulation_results.append({
            'odds_set': ', '.join(combination),
            'total_win_loss_amount': total_win_loss_amount,
            'total_capital_expended': total_capital_expended
        })
    simulation_results_df = pd.DataFrame(simulation_results)
    simulation_results_df['odds_set'] = simulation_results_df['odds_set'].astype(str)
    simulation_results_df['capital_diff'] = simulation_results_df['total_capital_expended'] - simulation_results_df['total_win_loss_amount']
    simulation_results_df['capital_diff'] = simulation_results_df['capital_diff'].clip(lower=0)
    simulation_results_df = simulation_results_df.sort_values(by='total_win_loss_amount', ascending=False).head(10)

    fig = px.bar(
        simulation_results_df,
        x='odds_set',
        y=['total_win_loss_amount'],
        labels={'odds_set': 'Odds Set', 'total_win_loss_amount': 'Total Win/Loss Amount'},
        title='What could be the best odds combination to maximize profits?',
        barmode='stack'
    )
    fig.for_each_trace(lambda trace: trace.update(name='Win/Loss Amount' if trace.name == 'total_win_loss_amount' else 'Capital'))
    fig.update_layout(xaxis_type='category')
    st.plotly_chart(fig)

    # Summary table grouped by tournament
    st.title('Performance per Tournament')
    summary_df = filtered_bet_df.groupby(['game', 'tournament', 'win_loss_code']).agg(
        win_loss_amount=('win_loss_amount', 'sum'),
        number_of_bets=('tournament', 'count'),
    ).reset_index()

    # Pivot the summary dataframe
    summary_df_pivot = summary_df.pivot_table(index=['game', 'tournament'], columns='win_loss_code', values='win_loss_amount', fill_value=0).reset_index()
    summary_df_pivot.columns.name = None  # Remove the axis name
    summary_df_pivot = summary_df_pivot.rename(columns={'WIN': 'win_amount', 'LOSS': 'loss_amount'})

    summary_df_final = summary_df_pivot.merge(
        summary_df.groupby(['game', 'tournament']).agg(
            number_of_bets=('number_of_bets', 'sum'),
        ).reset_index(),
        on=['game', 'tournament']
    )
    if 'win_amount' not in summary_df_final.columns:
        summary_df_final['win_amount'] = 0
    if 'loss_amount' not in summary_df_final.columns:
        summary_df_final['loss_amount'] = 0
        
    summary_df_final['net_profit'] = summary_df_final['win_amount'] + summary_df_final['loss_amount']
    summary_df_final = summary_df_final.sort_values(by=['net_profit'], ascending=False)
    st.dataframe(summary_df_final)

    # Bet history details table
    st.title('Bet Details')
    st.dataframe(bet_df)
