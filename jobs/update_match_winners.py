import pandas as pd
import requests
import os
import pprint
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
SEARCH_KEY = os.environ.get('GOOGLE_SEARCH_KEY')
ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID')
current_year = datetime.now().year
current_date = datetime.now()

def convert_to_datetime(date_str):
    return pd.to_datetime(f"{date_str} {current_year}", format='%b %d %Y')

def get_matches_completed():
    filepath = r'files/offline_logs'
    match_odds_df = pd.read_csv(f'{filepath}/match_odds_df.csv').set_index('id')
    match_odds_df['date'] = match_odds_df['date'].apply(convert_to_datetime)
    matches_completed_df = match_odds_df[match_odds_df['date']<current_date]
    matches_completed_df.sort_values(by=['game','date'],inplace=True)
    print(len(matches_completed_df))

def search_for_winner_of_match(query):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': query,
        'key': SEARCH_KEY,
        'cx': ENGINE_ID
    }

    response = requests.get(url, params=params)
    results = response.json()
    print([item['link'] for item in results['items']])

query='mlbb liquid-echo-vs-falcons-ap-bren July 12 2024'
search_for_winner_of_match(query)