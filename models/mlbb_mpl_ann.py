import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd

class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
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

def train_score_model_ann(X_train, X_test, y_train, y_test, epochs=100, batch_size=10):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor1 = torch.tensor(y_train['team1_score'].values, dtype=torch.long)
    y_train_tensor2 = torch.tensor(y_train['team2_score'].values, dtype=torch.long)
    y_test_tensor1 = torch.tensor(y_test['team1_score'].values, dtype=torch.long)
    y_test_tensor2 = torch.tensor(y_test['team2_score'].values, dtype=torch.long)
    input_size = X_train.shape[1]
    model = ANN(input_size)

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

def run_training_pipeline(training_file_folder):
    X_train = pd.read_csv(f'{training_file_folder}/X_train.csv')
    X_test = pd.read_csv(f'{training_file_folder}/X_test.csv')
    y_train = pd.read_csv(f'{training_file_folder}/y_train.csv')
    y_test = pd.read_csv(f'{training_file_folder}/y_test.csv')
    train_score_model_ann(X_train,X_test, y_train, y_test)

country = 'Philippines'
training_file_folder = f'files/mlbb/MPL/{country}/model_usage'
run_training_pipeline(training_file_folder)