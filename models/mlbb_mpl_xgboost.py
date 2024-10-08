import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def convert_df_to_arrays(X_train, X_test, y_train, y_test):
    X_train_array = X_train.values
    X_test_array = X_test.values
    y_train_array = y_train['winner'].values
    y_test_array = y_test['winner'].values
    X_column_orders = X_train.columns

    return X_train_array,X_test_array,y_train_array,y_test_array,X_column_orders

def train_model(X_train,X_test, y_train, y_test,country):
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Save model and SHAP explainer to pickle files
    with open(f'pickles/mpl/{country}/mlbb_model_xgboost.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open(f'pickles/mpl/{country}/mlbb_explainer_xgboost.pkl', 'wb') as f:
        pickle.dump(explainer, f)

    print("Model and SHAP explainer saved as pickle files.")

def load_inference_artifacts(filepath):
    with open(f'{filepath}/mlbb_model_xgboost.pkl', 'rb') as f:
        trained_model = pickle.load(f)

    with open(f'{filepath}/mpl_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(f'{filepath}/mpl_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return trained_model,label_encoder,scaler

def get_sample_data(df):
    sample = df.sample()
    return sample

def predict(trained_model,label_encoder,scaler, df):
    # df['year'] = label_encoder.transform(df['year'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # numeric_cols = numeric_cols.drop(['year'])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    features = df.values
    predictions = trained_model.predict(features)
    results = []
    for i in range(len(predictions)):
        prediction_result = {
            'prediction': int(predictions[i]),
            'features': features
        }
        results.append(prediction_result)
    
    return results

def get_sample_data(df):
    sample = df.sample()
    return sample

def run_training_pipeline(training_file_folder,country):
    X_train = pd.read_csv(f'{training_file_folder}/X_train.csv')
    X_test = pd.read_csv(f'{training_file_folder}/X_test.csv')
    y_train = pd.read_csv(f'{training_file_folder}/y_train.csv')
    y_test = pd.read_csv(f'{training_file_folder}/y_test.csv')
    X_train, X_test, y_train, y_test,X_column_orders =  convert_df_to_arrays(X_train, X_test, y_train, y_test)
    train_model(X_train,X_test, y_train, y_test,country)

### getting samples to test for inference
def run_sampling_for_inference(inference_file_folder):
    X_test = pd.read_csv(f'{inference_file_folder}/inference_samples.csv')
    X_test = X_test.drop('Unnamed: 0', axis=1)
    sample_df = get_sample_data(X_test)
    return sample_df

### inference for predicting results
def run_sample_inference(inference_file_path, df):
    trained_model,label_encoder,scaler = load_inference_artifacts(inference_file_path)
    prediction = predict(trained_model,label_encoder,scaler, df)
    return prediction


#### pipeline ###

if __name__=='__main__':
    start_train_model = False
    test_inference = True

    if start_train_model:
        country = 'Philippines'
        training_file_folder = f'files/mlbb/MPL/{country}/model_usage'
        run_training_pipeline(training_file_folder,country)

    if test_inference:
        country = 'Philippines'
        inference_file_folder = f'files/mlbb/MPL/{country}/model_usage'
        sample_df = run_sampling_for_inference(inference_file_folder)
        inference_file_path = f'pickles/mpl/{country}'
        prediction = run_sample_inference(inference_file_path, sample_df)
        print(prediction)

