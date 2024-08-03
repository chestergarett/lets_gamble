import pandas as pd

def check_valid_team_scores():
    file = f'files/mlbb/MPL/Philippines/model_usage/mpl_input_model_data.csv'
    df = pd.read_csv(file)
    print(df[df['team2_score']>2])

check_valid_team_scores()
    