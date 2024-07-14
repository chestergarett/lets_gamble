import requests
import glob
import os
import sys
import pandas as pd
from bs4 import BeautifulSoup
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firebase.connect import run_save_match_odds


class RivalryScraper:
    def __init__(self,url,game):
        self.response = requests.get(url)
        self.soup = BeautifulSoup(self.response.text, 'html.parser')
        self.divs = self.soup.find_all('div', class_='betline m-auto betline-wide mb-0')
        self.game = game
        self.matches = []

    def get_matches(self):
        for div in self.divs:
            right_team = div.select_one('button.left-facing-competitor .outcome-name').text.strip()
            right_team_odds = div.select_one('button.left-facing-competitor .outcome-odds').text.strip()
            draw_odds = div.select_one('button.middle-competitor .outcome-odds').text.strip() if div.select_one('button.middle-competitor .outcome-odds') else None
            left_team = div.select_one('button.right-facing-competitor .outcome-name').text.strip()
            left_team_odds = div.select_one('button.right-facing-competitor .outcome-odds').text.strip()
            date = div.select_one('strong').textß
            date = re.sub(r'\s+', ' ', date).strip() if date else None
            
            match_odds = {
                'right_team_name': right_team,
                'right_odds': right_team_odds,
                'left_team_name': left_team,
                'left_odds': left_team_odds,
                'draw_odds': draw_odds,
                'date': date,
                'winner': '',
                'game': self.game
            }
            
            self.matches.append(match_odds)

        return self.matches
                
def run_scraper_pipeline(games_dict,game_type):
    for game_key, game_url in games_dict.items():
        url = f"https://www.rivalry.com/{game_type}/{game_url}-betting"
        game = game_key
        scraper = RivalryScraper(url,game)
        matches = scraper.get_matches()
        print('matches',matches)
        run_save_match_odds(matches)
        print(f'Successfully scraped {game_key}')

esports_dict = {
    'wildrift': 'league-of-legends-wild-rift',
    'hok': 'honor-of-kings',
    'codm': 'call-of-duty-mobile',
    'starcraft': 'starcraft',
    'nba2k': 'nba2k',
    'r6': 'r6',
    'fc': 'fc',
    'cod': 'call-of-duty',
    'csgo': 'csgo',
    'valorant': 'valorant',
    'lol': 'league-of-legends',
    'dota2': 'dota-2',
}

sports_dict = {
    'boxing': 'boxing',
    'mma': 'mma'
}
    
run_scraper_pipeline(esports_dict,'esports')
run_scraper_pipeline(sports_dict, 'sports')