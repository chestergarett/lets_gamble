import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def parse_dates(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        return pd.to_datetime(date_str, format='%Y-%m-%d')

def clean_mlbb_msc_dataset(csv, country):
    df = pd.read_csv(csv)
    df = df.drop_duplicates()
    df['match_date'] = df['match_date'].apply(parse_dates)
    df['year'] = df['match_date'].dt.year
    ## add correct team1_game_wins,team1_game_losses,team2_match_wins,team2_match_losses,team2_game_wins,team2_game_losses
    df['team1_game_wins'] = (df['team1_game_wins'] - df['team1_score']).clip(lower=0)
    df['team1_game_losses'] = (df['team1_game_losses'] - df['team1_score']).clip(lower=0)
    df['team2_game_wins'] = (df['team2_game_wins'] - df['team2_score']).clip(lower=0)
    df['team2_game_losses'] = (df['team2_game_losses'] - df['team2_score']).clip(lower=0)
    df.loc[df['team1_score'] == 2, 'team1_match_wins'] = (df.loc[df['team1_score'] == 2, 'team1_match_wins'] - 1).clip(lower=0)
    df.loc[df['team1_score'] < 2, 'team1_match_losses'] = (df.loc[df['team1_score'] < 2, 'team1_match_losses'] - 1).clip(lower=0)
    df.loc[df['team2_score'] == 2, 'team2_match_wins'] = (df.loc[df['team2_score'] == 2, 'team2_match_wins'] - 1).clip(lower=0)
    df.loc[df['team2_score'] < 2, 'team2_match_losses'] = (df.loc[df['team2_score'] < 2, 'team2_match_losses'] - 1).clip(lower=0)

    df_swapped = df.copy()
    df_swapped[['team1', 'team2']] = df[['team2', 'team1']].values
    df_swapped[['team1_score', 'team2_score']] = df[['team2_score', 'team1_score']].values
    df_swapped[['team1_match_wins', 'team2_match_wins']] = df[['team2_match_wins', 'team1_match_wins']].values
    df_swapped[['team1_match_losses', 'team2_match_losses']] = df[['team2_match_losses', 'team1_match_losses']].values
    df_swapped[['team1_game_wins', 'team2_game_wins']] = df[['team2_game_wins', 'team1_game_wins']].values
    df_swapped[['team1_game_losses', 'team2_game_losses']] = df[['team2_game_losses', 'team1_game_losses']].values

    df_augmented = pd.concat([df, df_swapped], ignore_index=True)
    df_augmented['winner'] = df_augmented.apply(lambda row: 1 if row['team1_score'] > row['team2_score'] else 0, axis=1)
    df_augmented = df_augmented.sort_values(['match_date'])
    df_augmented = df_augmented.drop(['match_date', 'Unnamed: 0'],axis=1)
    print(df_augmented.columns)
    df_augmented.to_csv(f'files/mlbb/MPL/{country}/model_usage/mpl_input_model_data.csv', index=False)

def do_feature_engineering(df,country):
    df = df[(df['team2_score']<3) & (df['team1_score']<3)]
    X = df.drop(['team1', 'team2', 'team1_score', 'team2_score','winner'], axis=1)
    y = df[['team1_score', 'team2_score', 'winner']]
    label_encoder = LabelEncoder()
    X['year'] = label_encoder.fit_transform(X['year'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    filepath = r'files/mlbb/MPL'
    X_train.to_csv(f'{filepath}/{country}/model_usage/X_train.csv',index=False)
    X_test.to_csv(f'{filepath}/{country}/model_usage/X_test.csv',index=False)
    y_train.to_csv(f'{filepath}/{country}/model_usage/y_train.csv',index=False)
    y_test.to_csv(f'{filepath}/{country}/model_usage/y_test.csv',index=False)

    with open(f'pickles/mpl/{country}/mpl_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    with open(f'pickles/mpl/{country}/mpl_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print('Exported training and testing files')

def run_transformation_pipeline(csv,country):
    clean_mlbb_msc_dataset(csv,country)
    df = pd.read_csv(f'files/mlbb/MPL/{country}/model_usage/mpl_input_model_data.csv')
    do_feature_engineering(df,country)

country = 'Indonesia'
csv = f'files/mlbb/MPL/{country}/all_seasons.csv'
run_transformation_pipeline(csv,country)