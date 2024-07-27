import requests
import glob
import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

def parse_date(date_str):
    return datetime.strptime(date_str, '%B %d, %Y - %H:%M')

class ScrapeMatchInfo:
    def __init__(self,url):
        self.url = url
        self.response = requests.get(url)
        self.soup = BeautifulSoup(self.response.text, 'html.parser')
        self.weekly_matches = []
        self.weekly_matches_df = None
    def get_weekly_results(self):
        template_boxes = self.soup.find_all('div', class_='brkts-popup brkts-match-info-popup')
        for box in template_boxes:
            text_content = box.get_text(separator='|').split('|')
            data = {
                'team1': text_content[0],
                'team1_score': text_content[1],
                'team2': text_content[3],
                'team2_score': text_content[2],
                'match_date': text_content[4],
            }
            self.weekly_matches.append(data)
        
        self.weekly_matches_df = pd.DataFrame(self.weekly_matches)

    def transform_df(self):
        # Strip spaces and convert match_date to datetime
        self.weekly_matches_df['match_date'] = self.weekly_matches_df['match_date'].str.strip()
        self.weekly_matches_df['match_date'] = pd.to_datetime(self.weekly_matches_df['match_date'], format='%B %d, %Y - %H:%M')

        # Convert scores to numeric
        self.weekly_matches_df['team1_score'] = pd.to_numeric(self.weekly_matches_df['team1_score'])
        self.weekly_matches_df['team2_score'] = pd.to_numeric(self.weekly_matches_df['team2_score'])

        # Prepare dataframes for each team's scores
        df_team1 = self.weekly_matches_df[['team1', 'team1_score', 'team2_score', 'match_date']].rename(columns={'team1': 'team', 'team1_score': 'score', 'team2_score': 'opponent_score'})
        df_team2 = self.weekly_matches_df[['team2', 'team2_score', 'team1_score', 'match_date']].rename(columns={'team2': 'team', 'team2_score': 'score', 'team1_score': 'opponent_score'})

        # Concatenate dataframes
        df_scores = pd.concat([df_team1, df_team2])

        # Compute game wins and losses
        df_scores = df_scores.sort_values(by=['team', 'match_date'])
        df_scores['match_wins'] = (df_scores['score'] > df_scores['opponent_score']).astype(int)
        df_scores['match_losses'] = (df_scores['score'] < df_scores['opponent_score']).astype(int)

        # Calculate cumulative game wins and losses
        df_scores['game_wins'] = df_scores.groupby(['team'])['score'].cumsum()

        # Determine losses based on score difference
        df_scores['game_losses'] = df_scores['opponent_score'] - df_scores['score']
        df_scores['game_losses'] = df_scores['game_losses'].clip(lower=0)

        df_scores = df_scores.sort_values(by=['team', 'match_date'])
        df_scores['game_losses'] = df_scores.groupby(['team'])['game_losses'].cumsum()

        df_scores['match_wins'] = df_scores.groupby(['team'])['match_wins'].cumsum()
        df_scores['match_losses'] = df_scores.groupby(['team'])['match_losses'].cumsum()
        
        # Merge the cumulative losses back to df_scores
        df_scores = df_scores[['team', 'match_date', 'match_wins', 'match_losses', 'game_wins', 'game_losses']]
        print(df_scores[df_scores['team'] == 'AP.Bren'])


url = 'https://liquipedia.net/mobilelegends/MPL/Philippines/Season_13/Regular_Season'
scraper = ScrapeMatchInfo(url)
scraper.get_weekly_results()
scraper.transform_df()