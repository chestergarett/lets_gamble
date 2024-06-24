import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example data (replace with your actual dataset)
data = {
    'loser': ['Team A', 'Team B', 'Team C'],
    'loser_country': ['Country X', 'Country Y', 'Country Z'],
    'loser_group_rank': [1, 2, 3],
    'loser_group_win': [5, 4, 3],
    'loser_group_loss': [1, 2, 3],
    'loser_score': [100, 90, 80],
    'winner': ['Team B', 'Team C', 'Team A'],
    'winner_country': ['Country Y', 'Country Z', 'Country X'],
    'winner_group_rank': [2, 3, 1],
    'winner_group_win': [4, 3, 5],
    'winner_group_loss': [2, 3, 1],
    'winner_score': [90, 80, 100],
    'Year': [2023, 2023, 2023],
    'Tournament': ['Tournament A', 'Tournament A', 'Tournament A']
}

df = pd.DataFrame(data)

# Create pairwise dataset
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
pairs_df['team1_score'] = df['winner_score']
pairs_df['team2_score'] = df['loser_score']
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
pairs_df_reverse['team1_score'] = df['loser_score']
pairs_df_reverse['team2_score'] = df['winner_score']
pairs_df_reverse['outcome'] = 0  # Team 2 (winner) wins

# Concatenate both datasets
full_pairs_df = pd.concat([pairs_df, pairs_df_reverse], ignore_index=True)

# Feature columns
X = full_pairs_df.drop(['team1', 'team2', 'outcome'], axis=1)
y = full_pairs_df['outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example of predicting a new match pair (adjust features accordingly)
new_match = pd.DataFrame({
    'team1': ['Team C'],
    'team2': ['Team A'],
    'team1_country': ['Country Z'],
    'team2_country': ['Country X'],
    'team1_group_rank': [3],
    'team2_group_rank': [1],
    'team1_group_win': [3],
    'team2_group_win': [5],
    'team1_group_loss': [3],
    'team2_group_loss': [1],
    'team1_score': [80],
    'team2_score': [100]
})

# Predict the outcome
new_match_features = new_match.drop(['team1', 'team2'], axis=1)
prediction = model.predict(new_match_features)
if prediction[0] == 1:
    print(f"Predicted winner: {new_match['team1'][0]}")
else:
    print(f"Predicted winner: {new_match['team2'][0]}")
