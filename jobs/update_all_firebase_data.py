import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firebase.connect import edit_single_match,read_single_match,run_save_transactions_pipeline,query_blank_winners,edit_single_prediction,check_if_update_needed

def iterate_rows_of_df(df):
    for row in df.iterrows():
        row_id = row['id']
def upload_offline_to_firebase(df, df_type):
    if df_type=='match_odds':
        for index, row in df.iterrows():
            if row['winner']:
                id = row['id']
                update = {'winner': row['winner']}
                no_winner_posted = read_single_match(id)
                if no_winner_posted:
                    edit_single_match(id,update)
    elif df_type=='bet':
        new_bets_df = df[df['id'].str.startswith('nfb')]
        new_bets_df = new_bets_df.drop('id', axis=1)
        transactions = new_bets_df.to_dict(orient='records')
        run_save_transactions_pipeline(transactions)

        existing_bets_df = df[~df['id'].str.startswith('nfb')]
        for index,row in existing_bets_df.iterrows():
            check_if_update_needed(row['id'],row)
    elif df_type=='prediction':
        docs = query_blank_winners()
        if docs:
            for doc in docs:
                df = df[df['id']==doc.id]
                if not df.empty:
                    update = {'actual_winner': df['actual_winner'].values[0]}
                    edit_single_prediction(doc.id, update)

filepath=r'files/offline_logs/'
match_odds_df = pd.read_csv(f'{filepath}/match_odds_df.csv')
bet_df = pd.read_csv(f'{filepath}/bet_df.csv')
pred_df = pd.read_csv(f'{filepath}/pred_df.csv')


upload_offline_to_firebase(bet_df, 'bet')
