import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mlbb_mpl_xgboost import predict as predict_xgboost, load_inference_artifacts as load_inference_artifacts_xgboost
from models.mlbb_mpl_ann import load_inference_artifacts_ann as load_inference_artifacts_transformers

class TransformerWinner(nn.Module):
    def __init__(self, input_size, d_model=128, num_heads=4, num_layers=2, dim_feedforward=512, dropout=0.3):
        super(TransformerWinner, self).__init__()
        
        # Embedding layer to project input features to d_model dimensions
        self.embedding = nn.Linear(input_size, d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected output layer
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input embedding
        x = self.embedding(x).unsqueeze(0)
        x = x.permute(1, 0, 2)
        # Transformer encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over sequence length
        
        # Fully connected output
        x = self.fc_out(x)
        return x
    
def read_X_train_cols(country):
    df = pd.read_csv(f'files/mlbb/MPL/{country}/model_usage/X_train.csv')
    X_columns = list(df.columns)
    X_column_dict = {item: index for index, item in enumerate(X_columns)}
    return X_column_dict

def match_entry_data(data, X_column_dict):
    for_prediction_features = {key: 0 for key in X_column_dict.keys()}

    for key, value in data.items():
        if key in X_column_dict.keys():
            for_prediction_features[key] = value
        else:
            transformed_key = f'{key}_{value.title().strip()}'
            if transformed_key in X_column_dict.keys():
                for_prediction_features[transformed_key] = 1

    for_prediction_features = pd.DataFrame([for_prediction_features])
    return for_prediction_features

def predict_transformers(trained_model_config,label_encoder,scaler, df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    features = torch.tensor(df.values, dtype=torch.float32)
    
    model = TransformerWinner(features.shape[1])
    model.load_state_dict(torch.load(trained_model_config))
    model.eval()

    with torch.no_grad():
        predictions = model(features).numpy()
    predictions = (predictions > 0.5).astype(int).flatten()

    results = []
    for i in range(len(predictions)):
        prediction_result = {
            'prediction': int(predictions[i]),
            'features': features
        }
        results.append(prediction_result)
    
    return results


def do_xgboost_prediction(data,country):
    filepath = f'pickles/mpl/{country}'
    X_column_dict = read_X_train_cols(country)
    features_df = match_entry_data(data, X_column_dict)
    trained_model,label_encoder,scaler = load_inference_artifacts_xgboost(filepath)
    prediction = predict_xgboost(trained_model,label_encoder,scaler, features_df)
    predicted_winner = data['team1'] if prediction[0]['prediction']==1 else data['team2']
    return 'XGBoost', predicted_winner 

def do_transformers_prediction(data,country):
    filepath = f'pickles/mpl/{country}'
    X_column_dict = read_X_train_cols(country)
    features_df = match_entry_data(data, X_column_dict)
    trained_model_config,label_encoder,scaler = load_inference_artifacts_transformers(filepath)
    prediction = predict_transformers(trained_model_config,label_encoder,scaler, features_df)
    predicted_winner = data['team1'] if prediction[0]['prediction']==1 else data['team2']
    return 'Transformers', predicted_winner 

country = 'Indonesia'
data = {
        "team1": 'Rebellion',
        "team2": 'Geek Fam',
        "team1_match_wins": 6,
        "team1_match_losses": 9,
        "team1_game_wins": 16,
        "team1_game_losses": 14,
        "team2_match_wins": 9,
        "team2_match_losses": 6,
        "team2_game_wins": 20,
        "team2_game_losses": 8,
        "year": 2024
}

# prediction = do_xgboost_prediction(data, country)
prediction = do_transformers_prediction(data, country)
print(prediction)
