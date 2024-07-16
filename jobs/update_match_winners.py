import pandas as pd
import requests
import os
import pprint
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
SEARCH_KEY = os.environ.get('GOOGLE_SEARCH_KEY')
ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID')
current_year = datetime.now().year
current_date = datetime.now()

def convert_to_datetime(date_str):
    return pd.to_datetime(f"{date_str}", format='%Y-%m-%d')

def clean_winner_text(text):
    if pd.notna(text) and 'Winner!' in text:
        words = text.split()
        cleaned_text = ' '.join(words[1:]).replace('Winner!', '').strip()
        
        return cleaned_text
    return text

def get_matches_completed():
    filepath = r'files/offline_logs'
    match_odds_df = pd.read_csv(f'{filepath}/match_odds_df.csv').set_index('id')
    match_odds_df['date'] = match_odds_df['date'].apply(convert_to_datetime)
    matches_completed_df = match_odds_df[(match_odds_df['date']<current_date) & (match_odds_df['winner'].isna())]
    matches_completed_df.sort_values(by=['game','date'],inplace=True)

    return matches_completed_df,match_odds_df

def search_for_link_winner_of_match(query):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': query,
        'key': SEARCH_KEY,
        'cx': ENGINE_ID
    }

    response = requests.get(url, params=params)
    results = response.json()
    
    if 'items' in results.keys():
        links = [item['link'] for item in results['items']]
        return links[0]
    else:
        print(results)
        return None

def scrape_link_match_winner(link):
    texts = []
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    divs = soup.find_all('div', class_='match-team')

    for div in divs:
        full_text = ' '.join(div.stripped_strings)
        texts.append(full_text)
    
    return texts

def loop_each_row_and_get_winner(matches_completed_df):
    matches_completed_df['winner'] = None

    for index, row in matches_completed_df.iterrows():
        game = row['game'] 
        date = row['formatted_date'] 
        left_team_name = row['left_team_name'].replace(' ', '-')
        right_team_name = row['right_team_name'].replace(' ', '-')
        query = f'{game} {left_team_name}-vs-{right_team_name} {date}'
        link = search_for_link_winner_of_match(query)
        print('link', link)
        if link:
            texts = scrape_link_match_winner(link)
            winner_found = False
            for text in texts:
                if 'winner' in text.lower():
                    matches_completed_df.at[index, 'winner'] = clean_winner_text(text)
                    winner_found = True  
            if not winner_found: 
                matches_completed_df.at[index, 'winner'] = 'Draw'

            break   
        else:
            print(query, link)
            break

    return matches_completed_df

def update_match_odds_df(match_odds_df, matches_completed_df):
    filepath = r'files/offline_logs'
    matches_completed_df['winner'] = matches_completed_df['winner'].apply(clean_winner_text)
    match_odds_df = match_odds_df.reset_index()
    matches_completed_df = matches_completed_df.reset_index()
    merged_df = match_odds_df.merge(matches_completed_df[['id', 'winner']],
                                on=['id'],
                                how='left')
    
    match_odds_df['winner'] = match_odds_df['winner'].where(match_odds_df['winner'].notnull(), merged_df['winner_y'])
    match_odds_df = match_odds_df.loc[:, ~match_odds_df.columns.str.startswith('Unnamed:')]
    match_odds_df.to_csv(f'{filepath}/match_odds_df.csv')

def scrape_matches_completed_winners():
    matches_completed_df,match_odds_df = get_matches_completed()
    matches_completed_df['formatted_date'] = matches_completed_df['date'].dt.strftime('%B %d %Y')
    matches_completed_df = loop_each_row_and_get_winner(matches_completed_df)
    matches_completed_df.to_csv(f'files/offline_logs/matches_completed_df.csv')
    update_match_odds_df(match_odds_df, matches_completed_df)

def merge_offline_completed_winners():
    filepath = r'files/offline_logs'
    match_odds_df =pd.read_csv(f'{filepath}/match_odds_df.csv')
    matches_completed_df =pd.read_csv(f'{filepath}/matches_completed_df.csv')
    update_match_odds_df(match_odds_df, matches_completed_df)

scrape_matches_completed_winners()