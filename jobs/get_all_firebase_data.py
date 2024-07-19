import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firebase.connect import run_firebase_pipeline

def save_offline_firebase_logs():
    filepath = r'files/offline_logs'
    bet_df = run_firebase_pipeline('bet_logs')
    match_odds_df = run_firebase_pipeline('match_logs')
    pred_df = run_firebase_pipeline('model_prediction_logs')
    
    bet_df.to_csv(f'{filepath}/bet_df.csv')
    match_odds_df.to_csv(f'{filepath}/match_odds_df.csv')
    pred_df.to_csv(f'{filepath}/pred_df.csv')

def load_offline_df(data_type):
    filepath = r'files/offline_logs'
    if data_type=='bet_logs':
        bet_df = pd.read_csv(f'{filepath}/bet_df.csv').set_index('id')
        return bet_df
    if data_type=='model_prediction_logs':
        pred_df = pd.read_csv(f'{filepath}/pred_df.csv').set_index('id')
        return pred_df
    if data_type=='match_logs':
        match_odds_df = pd.read_csv(f'{filepath}/match_odds_df.csv').set_index('id')
        return match_odds_df
    

save_online_data = False

if save_online_data:
    save_offline_firebase_logs()