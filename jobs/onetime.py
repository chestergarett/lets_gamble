import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firebase.connect import edit_single_bet

filepath= f'files/offline_logs'

def update_all_bet_dates_in_firebase():
    bet_df = pd.read_csv(f'{filepath}/bet_df.csv').set_index('id')
    for index,row in bet_df.iterrows():
        update = {'bet_date': row['bet_date']}
        edit_single_bet(index,update)

update_all_bet_dates_in_firebase()