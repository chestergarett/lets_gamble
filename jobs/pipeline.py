import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jobs.get_all_firebase_data import save_offline_firebase_logs
from jobs.update_match_winners import scrape_matches_completed_winners
from jobs.update_all_firebase_data import upload_offline_to_firebase

def run_job_pipeline(code):
    filepath=r'files/offline_logs/'
    match_odds_df = pd.read_csv(f'{filepath}/match_odds_df.csv')
    bet_df = pd.read_csv(f'{filepath}/bet_df.csv')
    pred_df = pd.read_csv(f'{filepath}/pred_df.csv')
    
    if code==0:
        scrape_matches_completed_winners()
        upload_offline_to_firebase(match_odds_df, 'match_odds')
        upload_offline_to_firebase(bet_df, 'bet_df')
        upload_offline_to_firebase(pred_df, 'pred')
    if code==1:
        save_offline_firebase_logs()
    if code==2:
        scrape_matches_completed_winners()
    if code==3:
        upload_offline_to_firebase(match_odds_df, 'match_odds')
    if code==4:
        upload_offline_to_firebase(bet_df, 'bet_df')
    if code==5:
        upload_offline_to_firebase(pred_df, 'pred')
    if code==6:
        upload_offline_to_firebase(bet_df, 'bet_df')
        upload_offline_to_firebase(pred_df, 'pred')
        upload_offline_to_firebase(match_odds_df, 'match_odds')
    if code==7:
        save_offline_firebase_logs()
        scrape_matches_completed_winners()


run_job_pipeline(1)
