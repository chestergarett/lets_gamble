import requests
from bs4 import BeautifulSoup


class ScrapeGameInfo:
    def __init__(self,url):
        self.url = url
        self.response = requests.get(url)
        self.soup = BeautifulSoup(self.response.text, 'html.parser')

    def get_group_stage_rankings(self):
        tables = self.soup.find_all('table', class_='wikitable wikitable-bordered')
        groups = []
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
                        group.append({'rank': rank, 'team': team, 'match': match})
        
                groups.append(group)

        print(groups)
            

    def get_playoffs_results(soup):
        playoffs_section = soup.find('span', id="Playoffs").parent.find_next('div', class_="bracket").find_all('div', class_="brkts-matchlist")
        winners = []
        losers = []

        for match in playoffs_section:
            teams = match.find_all('span', class_="teamname")
            scores = match.find_all('span', class_="team-template-text")

            if len(teams) == 2 and len(scores) == 2:
                team1 = teams[0].get_text(strip=True)
                team2 = teams[1].get_text(strip=True)
                score1 = int(scores[0].get_text(strip=True))
                score2 = int(scores[1].get_text(strip=True))

                if score1 > score2:
                    winners.append(team1)
                    losers.append(team2)
                else:
                    winners.append(team2)
                    losers.append(team1)

        return winners, losers

url = "https://liquipedia.net/mobilelegends/MSC/2021"
msc_2022 = ScrapeGameInfo(url)
msc_2022.get_group_stage_rankings()