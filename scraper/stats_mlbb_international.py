import requests
import glob
import os
import pandas as pd
from bs4 import BeautifulSoup

class ScrapeGameInfo:
    def __init__(self,url, tournament):
        self.url = url
        self.year = int(url.split('/')[-1])
        self.tournament = tournament
        self.response = requests.get(url)
        self.soup = BeautifulSoup(self.response.text, 'html.parser')
        self.groups = []

        self.playoffs_url = f'{url}/Playoffs'  if int(url.split('/')[-1]) < 2023 else f'{url}/Knockout_Stage'
        self.playoffs_response = requests.get(url)
        self.playoffs_soup = BeautifulSoup(self.playoffs_response.text, 'html.parser')
        self.playoff_matches = []
        self.playoff_df = None


    def get_group_stage_rankings(self):
        tables = self.soup.find_all('table', class_='wikitable wikitable-bordered')
        for index, table in enumerate(tables):
            group = []
            tbody = table.find('tbody')
            
            if 'Group' in tbody.get_text():
                headers = table.find_all('th')
                header_texts = [header.get_text(strip=True) for header in headers]

                rows = table.find_all('tr')[1:]
                for rank,row in enumerate(rows):
                    columns = row.find_all('td')
                    if len(columns) > 1:
                        team = columns[0].get_text(strip=True)
                        match = columns[1].get_text(strip=True)
                        country = get_team_country(team)
                        group.append({'rank': rank, 'team': team, 'match': match, 'country': country})
        
                self.groups.append(group)

    def get_playoffs_results(self):
        brackets = self.playoffs_soup.find_all('div', class_='brkts-match-has-details')
        matches = []
        for bracket in brackets:
            opponents = bracket.find_all('div', class_='brkts-opponent-entry')
            match = []
            for opponent in opponents:
                highlighting_span = opponent.find('span', attrs={'data-highlightingclass': True})
                if highlighting_span:
                    team_name = highlighting_span['data-highlightingclass']
                
                score_div = opponent.find('div', class_='brkts-opponent-score-inner')
                if score_div:
                    score = score_div.get_text(strip=True)

                match.append({'team': team_name, 'score': score})
            
            self.playoff_matches.append(match)
        df = convert_matches_to_df(self.playoff_matches, self.groups)
        self.playoff_df = df
        self.playoff_df['Year'] = self.year
        self.playoff_df['Tournament'] = self.tournament
        self.playoff_df.to_csv(f'files/mlbb/{self.tournament}/{self.year}.csv')
        print(f'Successfully scraped and exported data for {self.tournament} {self.year}')

def convert_matches_to_df(matches, groups):
    rows = []
    for match in matches:
        team1, team2 = match
        if int(team1['score']) > int(team2['score']):
            winner, loser = team1, team2
        else:
            winner, loser = team2, team1

        winner_group_rank, winner_group_win, winner_group_loss, winner_country = get_group_stats(winner, groups)
        loser_group_rank, loser_group_win, loser_group_loss, loser_country = get_group_stats(loser, groups)
        rows.append({
            'loser': loser['team'],
            'loser_country': loser_country,
            'loser_group_rank': loser_group_rank,
            'loser_group_win': loser_group_win,
            'loser_group_loss': loser_group_loss,
            'loser_score': int(loser['score']),
            'winner': winner['team'],
            'winner_country': winner_country,
            'winner_group_rank': winner_group_rank,
            'winner_group_win': winner_group_win,
            'winner_group_loss': winner_group_loss,
            'winner_score': int(winner['score'])
        })
   
    df = pd.DataFrame(rows)
    return df

def get_team_country(team):
        team_updated = team.replace(' ', '_')
        url = f'https://liquipedia.net/mobilelegends/{team_updated}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        flag_span = soup.find('span', class_='flag')
        if flag_span:
            a_tag = flag_span.find('a')
            if a_tag and 'title' in a_tag.attrs:
                country = a_tag['title']
                return country

def get_group_stats(team, groups):
    flattened_groups = [team for group in groups for team in group]
    print(team, flattened_groups)
    for group_team in flattened_groups:
        ### cleanup dirty data from web ###
        if team['team'].lower() == 'aerowolf pro team':
            team['team'] = 'aerowolf roxy'
        elif team['team'].lower() == 'rrq hoshi':
            team['team'] = 'rrq.o2'

        if group_team['team'].lower()==team['team'].lower():
            group_rank = group_team['rank']
            group_win, group_loss = group_team['match'].split('-')[0], group_team['match'].split('-')[1]
            group_country = group_team['country']

            return group_rank, group_win, group_loss,group_country
        
def loop_per_year_then_scrape(years):
    for year in years:
        url = f"https://liquipedia.net/mobilelegends/MSC/{year}"
        msc = ScrapeGameInfo(url, 'MSC')
        msc.get_group_stage_rankings()
        msc.get_playoffs_results()

def concatenate_dfs(folder_path):
    all_files = glob.glob(os.path.join(folder_path, '*.csv'))
    dfs = []
    for file in all_files:
        df = pd.read_csv(file,index_col=None)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(f'{folder_path}/all_years.csv', index=False)

def get_country_head_to_head(csv):
    df = pd.read_csv(csv)
    grouped_df = df.groupby(['Year','winner_country', 'loser_country']).size().reset_index()
    grouped_df.to_csv(r'files/mlbb/MSC/aggregates/country_matchup.csv')

### scrape per year MSC results
years = [2024]
loop_per_year_then_scrape(years)

#### contenate per year stats into one dataframe
folder_path = r'files/mlbb/MSC/'
concatenate_dfs(folder_path)

### country head to head matchups
concatenated_csv = r'files/mlbb/MSC/all_years.csv'
get_country_head_to_head(concatenated_csv)