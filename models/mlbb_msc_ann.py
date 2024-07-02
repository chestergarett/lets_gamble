import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import shap
import pickle
import pandas as pd

class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x =  self.fc_layers(x)
        return x
    
def train_model_ann(X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    input_size = X_train.shape[1]
    model = ANNModel(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_class = (y_pred > 0.5).float()

    accuracy = accuracy_score(y_test, y_pred_class.numpy())
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred_class.numpy())
    print(f"Confusion Matrix:\n{cm}")

    explainer = shap.DeepExplainer(model, X_train_tensor)
    print('explainer', explainer)
    shap_values = explainer.shap_values(X_test_tensor)

    with open('pickles/mlbb_international_model_ann.pkl', 'wb') as f:
        pickle.dump(model.state_dict(), f)

    with open('pickles/mlbb_international_explainer_ann.pkl', 'wb') as f:
        pickle.dump(explainer, f)

    print("Model and SHAP explainer saved as pickle files.")

def convert_df_to_arrays(X_train, X_test, y_train, y_test):
    X_train_array = X_train.values
    X_test_array = X_test.values
    y_train_array = y_train.values
    y_test_array = y_test.values
    X_column_orders = X_train.columns

    return X_train_array,X_test_array,y_train_array,y_test_array,X_column_orders

def load_inference_artifacts_ann(filepath):
    with open(f'{filepath}/mlbb_international_model_ann.pkl', 'rb') as f:
        trained_model = pickle.load(f)

    with open(f'{filepath}/mlbb_international_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(f'{filepath}/mlbb_international_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f'{filepath}/mlbb_international_explainer_ann.pkl', 'rb') as f:
        explainer = pickle.load(f)
    
    return trained_model,label_encoder,scaler,explainer

def predict_ann(trained_model, trained_model_state_dict,label_encoder,scaler,explainer, df):
    df['year'] = label_encoder.fit_transform(df['year'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    features = torch.tensor(df.values, dtype=torch.float32)
    
    trained_model.load_state_dict(trained_model_state_dict)
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(features).numpy()
    predictions = (predictions > 0.5).astype(int).flatten()
    shap_values = explainer.shap_values(features)
    results = []
    for i in range(len(predictions)):
        explanation = dict(zip(df.columns, shap_values[i].reshape(1,-1)[0]))
        top_3_explanation = dict(sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)[:3])
        prediction_result = {
            'prediction': int(predictions[i]),
            'explanation': top_3_explanation
        }
        results.append(prediction_result)
    
    return results

def get_sample_data(df):
    sample = df.sample()
    return sample

def run_sampling_for_inference(inference_file_folder):
    X_test = pd.read_csv(f'{inference_file_folder}/X_test.csv')
    sample_df = get_sample_data(X_test)
    return sample_df

def run_sample_inference(inference_file_path, df):
    trained_model,label_encoder,scaler,explainer = load_inference_artifacts_ann(inference_file_path)
    predict_ann(trained_model,label_encoder,scaler,explainer, df)

def run_training_pipeline(training_file_folder):
    X_train = pd.read_csv(f'{training_file_folder}/X_train.csv')
    X_test = pd.read_csv(f'{training_file_folder}/X_test.csv')
    y_train = pd.read_csv(f'{training_file_folder}/y_train.csv')
    y_test = pd.read_csv(f'{training_file_folder}/y_test.csv')
    X_train, X_test, y_train, y_test,X_column_orders =  convert_df_to_arrays(X_train, X_test, y_train, y_test)
    train_model_ann(X_train,X_test, y_train, y_test)

#### pipeline ###
start_train_model = True
test_inference = False

if start_train_model:
    training_file_folder = r'files/mlbb/MSC/model_usage'
    run_training_pipeline(training_file_folder)

if test_inference:
    inference_file_folder = r'files/mlbb/MSC/model_usage'
    sample_df = run_sampling_for_inference(inference_file_folder)
    inference_file_path = r'pickles/'
    run_sample_inference(inference_file_path, sample_df)