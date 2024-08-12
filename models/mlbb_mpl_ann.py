import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd

class ANN_score(nn.Module):
    def __init__(self, input_size):
        super(ANN_score, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer for 3 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x1 = self.fc3(x)  # Separate outputs for team1 and team2
        x2 = self.fc3(x)
        return x1, x2


def adjust_predictions(y_pred):
    adjusted_preds = []

    for t1_score, t2_score in y_pred:
        if t1_score == 2:
            t2_score = min(t2_score, 1)  # Ensuring t2_score is either 1 or 0
        elif t2_score == 2:
            t1_score = min(t1_score, 1)  # Ensuring t1_score is either 1 or 0
        adjusted_preds.append([t1_score, t2_score])
    
    return np.array(adjusted_preds)

class TransformerWinner(nn.Module):
    def __init__(self, input_size, d_model=128, num_heads=4, num_layers=2, dim_feedforward=512, dropout=0.3):
        super(TransformerWinner, self).__init__()
        
        # Embedding layer to project input features to d_model dimensions
        self.embedding = nn.Linear(input_size, d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
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
        # Transformer encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Average over sequence length
        
        # Fully connected output
        x = self.fc_out(x)
        return x

def train_score_model_ann(X_train, X_test, y_train, y_test, epochs=200, batch_size=10):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor1 = torch.tensor(y_train['team1_score'].values, dtype=torch.long)
    y_train_tensor2 = torch.tensor(y_train['team2_score'].values, dtype=torch.long)
    y_test_tensor1 = torch.tensor(y_test['team1_score'].values, dtype=torch.long)
    y_test_tensor2 = torch.tensor(y_test['team2_score'].values, dtype=torch.long)
    input_size = X_train.shape[1]
    model = ANN_score(input_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs1, outputs2 = model(X_train_tensor)
        loss1 = criterion(outputs1, y_train_tensor1)
        loss2 = criterion(outputs2, y_train_tensor2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'mlbb_ann_score_model.pth')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs1, outputs2 = model(X_test_tensor)
        _, predicted1 = torch.max(outputs1, 1)
        _, predicted2 = torch.max(outputs2, 1)
    
    accuracy1 = accuracy_score(y_test_tensor1, predicted1)
    accuracy2 = accuracy_score(y_test_tensor2, predicted2)
    print(f'Accuracy for team1_score: {accuracy1}, team2_score: {accuracy2}')
    
    return model

def train_winner_model_transformer(X_train, X_test, y_train, y_test, country, epochs=200, batch_size=10):
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train['winner'].values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test['winner'].values, dtype=torch.float32).view(-1, 1)
    
    input_size = X_train.shape[1]
    model = TransformerWinner(input_size)  # Use the TransformerWinner model

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Transformer model expects input shape [sequence length, batch size, feature size]
        outputs = model(X_train_tensor)  # Add a sequence length dimension
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), f'pickles/mpl/{country}/mlbb_model_transformer.pth')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)  # Add a sequence length dimension
        predicted = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
    
    accuracy = accuracy_score(y_test_tensor, predicted)
    print('predicted', predicted)
    print(f'Accuracy for winning team: {accuracy:.4f}')
    
    return model

def run_training_pipeline(training_file_folder,country):
    X_train = pd.read_csv(f'{training_file_folder}/X_train.csv')
    X_test = pd.read_csv(f'{training_file_folder}/X_test.csv')
    y_train = pd.read_csv(f'{training_file_folder}/y_train.csv')
    y_test = pd.read_csv(f'{training_file_folder}/y_test.csv')
    # train_score_model_ann(X_train,X_test, y_train, y_test)
    train_winner_model_transformer(X_train,X_test, y_train, y_test,country)

def get_sample_data(df):
    sample = df.sample()
    return sample

def run_sampling_for_inference(inference_file_folder):
    X_test = pd.read_csv(f'{inference_file_folder}/inference_samples.csv')
    X_test = X_test.drop('Unnamed: 0', axis=1)
    sample_df = get_sample_data(X_test)
    return sample_df

def load_inference_artifacts_ann(filepath):
    trained_model = f'{filepath}/mlbb_model_transformer.pth'

    with open(f'{filepath}/mpl_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(f'{filepath}/mpl_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return trained_model,label_encoder,scaler

def predict_winner(trained_model_config,label_encoder,scaler, df):
    # df['year'] = label_encoder.transform(df['year'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # numeric_cols = numeric_cols.drop(['year'])
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

def run_sample_inference(inference_file_path, df):
    trained_model,label_encoder,scaler = load_inference_artifacts_ann(inference_file_path)
    prediction = predict_winner(trained_model,label_encoder,scaler, df)
    print(prediction)

#### pipeline ###
start_train_model = False
test_inference = False

if start_train_model:
    country = 'Indonesia'
    training_file_folder = f'files/mlbb/MPL/{country}/model_usage'
    run_training_pipeline(training_file_folder,country)

if test_inference:
    country = 'Indonesia'
    inference_file_folder = f'files/mlbb/MPL/{country}/model_usage'
    sample_df = run_sampling_for_inference(inference_file_folder)
    inference_file_path = f'pickles/mpl/{country}'
    run_sample_inference(inference_file_path, sample_df)