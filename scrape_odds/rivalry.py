import requests
import glob
import os
import pandas as pd
from bs4 import BeautifulSoup

url = f"https://www.rivalry.com/esports/honor-of-kings-betting"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
divs = soup.find_all('div', class_='bet-center-content-markets')

for div in divs:
    right_team = soup.select_one('button.left-facing-competitor .outcome-name').text.strip()
    right_team_odds = soup.select_one('button.left-facing-competitor .outcome-odds').text.strip()
    draw_odds = soup.select_one('button.middle-competitor .outcome-odds').text.strip() if soup.select_one('button.middle-competitor .outcome-odds') else None
    left_team = soup.select_one('button.right-facing-competitor .outcome-name').text.strip()
    left_team_odds = soup.select_one('button.right-facing-competitor .outcome-odds').text.strip()
    
    match_odds = {
        'right_team_name': right_team,
        'right_odds': right_team_odds,
        'left_team_name': left_team,
        'left_odds': left_team_odds,
        'draw_odds': draw_odds
    }

    print('match_odds', match_odds)
    