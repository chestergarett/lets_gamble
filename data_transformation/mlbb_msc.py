import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def clean_mlbb_msc_dataset(csv):
    df = pd.read_csv(csv)
    columns_to_drop = ['Unnamed: 0', 'Unnamed: 0.1']
    df = df.drop(columns=columns_to_drop)
    df = df.drop_duplicates()

    pairs_df = pd.DataFrame()
    # Team 1 as winner, Team 2 as loser
    pairs_df['team1'] = df['winner']
    pairs_df['team2'] = df['loser']
    pairs_df['team1_country'] = df['winner_country']
    pairs_df['team2_country'] = df['loser_country']
    pairs_df['team1_group_rank'] = df['winner_group_rank']
    pairs_df['team2_group_rank'] = df['loser_group_rank']
    pairs_df['team1_group_win'] = df['winner_group_win']
    pairs_df['team2_group_win'] = df['loser_group_win']
    pairs_df['team1_group_loss'] = df['winner_group_loss']
    pairs_df['team2_group_loss'] = df['loser_group_loss']
    pairs_df['year'] = df['Year']
    pairs_df['tournament'] = df['Tournament']
    pairs_df['outcome'] = 1  # Team 1 (winner) wins

    # Team 1 as loser, Team 2 as winner (reverse perspective)
    pairs_df_reverse = pd.DataFrame()
    pairs_df_reverse['team1'] = df['loser']
    pairs_df_reverse['team2'] = df['winner']
    pairs_df_reverse['team1_country'] = df['loser_country']
    pairs_df_reverse['team2_country'] = df['winner_country']
    pairs_df_reverse['team1_group_rank'] = df['loser_group_rank']
    pairs_df_reverse['team2_group_rank'] = df['winner_group_rank']
    pairs_df_reverse['team1_group_win'] = df['loser_group_win']
    pairs_df_reverse['team2_group_win'] = df['winner_group_win']
    pairs_df_reverse['team1_group_loss'] = df['loser_group_loss']
    pairs_df_reverse['team2_group_loss'] = df['winner_group_loss']
    pairs_df_reverse['year'] = df['Year']
    pairs_df_reverse['tournament'] = df['Tournament']
    pairs_df_reverse['outcome'] = 0  # Team 2 (winner) wins

    # Concatenate both datasets
    full_pairs_df = pd.concat([pairs_df, pairs_df_reverse], ignore_index=True)

    full_pairs_df.to_csv(r'files/mlbb/MSC/model_usage/msc_input_model_data.csv', index=False)

def do_feature_engineering(df):
    X = df.drop(['team1', 'team2', 'outcome'], axis=1)
    y = df['outcome']
    X = pd.get_dummies(X, columns=['team1_country', 'team2_country', 'tournament'])
    X = X.astype(int)
    label_encoder = LabelEncoder()
    X['year'] = label_encoder.fit_transform(X['year'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    X_train.to_csv(r'files/mlbb/MSC/model_usage/X_train.csv',index=False)
    X_test.to_csv(r'files/mlbb/MSC/model_usage/X_test.csv',index=False)
    y_train.to_csv(r'files/mlbb/MSC/model_usage/y_train.csv',index=False)
    y_test.to_csv(r'files/mlbb/MSC/model_usage/y_test.csv',index=False)

    with open(r'pickles/mlbb_international_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    with open(r'pickles/mlbb_international_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print('Exported training and testing files')

def run_transformation_pipeline(csv):
    clean_mlbb_msc_dataset(csv)
    df = pd.read_csv(r'files/mlbb/MSC/model_usage/msc_input_model_data.csv')
    do_feature_engineering(df)

csv = r'files/mlbb/MSC/all_years.csv'
run_transformation_pipeline(csv)