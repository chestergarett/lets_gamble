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
    y_train_array = y_train.values
    y_test_array = y_test.values
    X_column_orders = X_train.columns

    return X_train_array,X_test_array,y_train_array,y_test_array,X_column_orders

def train_model(X_train,X_test, y_train, y_test):
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
    with open(r'pickles/mlbb_international_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open(r'pickles/mlbb_international_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)

    print("Model and SHAP explainer saved as pickle files.")

def load_inference_artifacts(filepath):
    with open(f'{filepath}/mlbb_international_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)

    with open(f'{filepath}/mlbb_international_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(f'{filepath}/mlbb_international_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f'{filepath}/mlbb_international_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    
    return trained_model,label_encoder,scaler,explainer

def get_sample_data(df):
    sample = df.sample()
    print('sample',sample)
    return sample

def predict(trained_model,label_encoder,scaler,explainer, df):
    df['year'] = label_encoder.fit_transform(df['year'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    features = df.values
    predictions = trained_model.predict(features)
    shap_values = explainer(features)

    results = []
    for i in range(len(predictions)):
        explanation = dict(zip(df.columns, shap_values[i].values))
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

def run_training_pipeline(training_file_folder):
    X_train = pd.read_csv(f'{training_file_folder}/X_train.csv')
    X_test = pd.read_csv(f'{training_file_folder}/X_test.csv')
    y_train = pd.read_csv(f'{training_file_folder}/y_train.csv')
    y_test = pd.read_csv(f'{training_file_folder}/y_test.csv')
    X_train, X_test, y_train, y_test,X_column_orders =  convert_df_to_arrays(X_train, X_test, y_train, y_test)
    train_model(X_train,X_test, y_train, y_test)

### getting samples to test for inference
def run_sampling_for_inference(inference_file_folder):
    X_test = pd.read_csv(f'{inference_file_folder}/X_test.csv')
    sample_df = get_sample_data(X_test)
    return sample_df

### inference for predicting results
def run_sample_inference(inference_file_path, df):
    trained_model,label_encoder,scaler,explainer = load_inference_artifacts(inference_file_path)
    prediction = predict(trained_model,label_encoder,scaler,explainer, df)
    print(prediction)


if __name__=='__main__':
    #### pipeline ###
    start_train_model = False
    test_inference = True

    if start_train_model:
        training_file_folder = r'files/mlbb/MSC/model_usage'
        run_training_pipeline(training_file_folder)

    if test_inference:
        inference_file_folder = r'files/mlbb/MSC/model_usage'
        sample_df = run_sampling_for_inference(inference_file_folder)
        inference_file_path = r'pickles/'
        run_sample_inference(inference_file_path, sample_df)

