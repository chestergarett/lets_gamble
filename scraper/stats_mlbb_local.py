import requests
import glob
import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%B %d, %Y - %H:%M')
    except ValueError:
        try:
            return datetime.strptime(date_str, '%B %d %Y')
        except ValueError:
            # Handle other formats or raise an error
            raise ValueError(f"Date format for '{date_str}' not recognized.")

class ScrapeMatchInfo:
    def __init__(self,url):
        self.url = url
        self.response = requests.get(url)
        self.soup = BeautifulSoup(self.response.text, 'html.parser')
        self.weekly_matches = []
        self.weekly_matches_df = None
        print('url',url)
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

    def transform_df(self,country,season):
        # Strip spaces and convert match_date to datetime
        self.weekly_matches_df['match_date'] = self.weekly_matches_df['match_date'].str.strip()
        try:
            self.weekly_matches_df['match_date'] = pd.to_datetime(self.weekly_matches_df['match_date'], format='%B %d, %Y - %H:%M')
        except ValueError:
            self.weekly_matches_df['match_date'] = pd.to_datetime(self.weekly_matches_df['match_date'], format='%B %d, %Y')

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
        team1_scores = df_scores.rename(columns={
        'team': 'team1', 
        'match_wins': 'team1_match_wins', 
        'match_losses': 'team1_match_losses', 
        'game_wins': 'team1_game_wins', 
        'game_losses': 'team1_game_losses'
        })
        
        team2_scores = df_scores.rename(columns={
            'team': 'team2', 
            'match_wins': 'team2_match_wins', 
            'match_losses': 'team2_match_losses', 
            'game_wins': 'team2_game_wins', 
            'game_losses': 'team2_game_losses'
        })

        self.weekly_matches_df = self.weekly_matches_df.merge(
            team1_scores,
            how='left',
            left_on=['team1', 'match_date'],
            right_on=['team1', 'match_date']
        )

        self.weekly_matches_df = self.weekly_matches_df.merge(
            team2_scores,
            how='left',
            left_on=['team2', 'match_date'],
            right_on=['team2', 'match_date']
        )

        self.weekly_matches_df.to_csv(f'files/mlbb/MPL/{country}/season{season}.csv')
        return self.weekly_matches_df
        

def scrape_per_year(base_url,country,seasons):
    for season in seasons:
        full_url = f'{base_url}/{country}/Season_{season}/Regular_Season'
        scraper = ScrapeMatchInfo(full_url)
        scraper.get_weekly_results()
        scraper.transform_df(country,season)
        print(f'Scraped MPL {country} Season {season} stats and saved to dataframe')

def concat_all_df(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, 'season*.csv'))
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.str.startswith('Unnamed')]
    combined_df.to_csv(f'{folder_path}/all_seasons.csv')

base_url = 'https://liquipedia.net/mobilelegends/MPL'
country = 'Indonesia'
seasons = [13,12,11,10,9,8,7,6,5,4,3,2,1]
folder_path = f'files/mlbb/MPL/Indonesia'

scrape_per_year(base_url,country,seasons)
concat_all_df(folder_path)