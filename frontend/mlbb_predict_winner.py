import streamlit as st
import pandas as pd
import os
from models.mlbb_msc_xgboost import predict, load_inference_artifacts
from models.mlbb_msc_ann import predict_ann, load_inference_artifacts_ann
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim

class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x =  self.fc_layers(x)
        return x
        
def app():
    st.title('MLBB predict winner')
    
    # Create columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Team 1")
        team1 = st.text_input("Team 1")
        team1_country = st.text_input("Team 1 Country")
        team1_group_rank = st.number_input("Team 1 Group Rank", min_value=1, step=1)
        team1_group_win = st.number_input("Team 1 Group Wins", min_value=0, step=1)
        team1_group_loss = st.number_input("Team 1 Group Losses", min_value=0, step=1)
    
    with col2:
        st.header("Team 2")
        team2 = st.text_input("Team 2")
        team2_country = st.text_input("Team 2 Country")
        team2_group_rank = st.number_input("Team 2 Group Rank", min_value=1, step=1)
        team2_group_win = st.number_input("Team 2 Group Wins", min_value=0, step=1)
        team2_group_loss = st.number_input("Team 2 Group Losses", min_value=0, step=1)
    
    # Year and Tournament fields spanning both columns
    year = st.number_input("Year", min_value=2017, max_value=2024, step=1)
    tournament = st.selectbox("Tournament", options=["MSC", "Other"])
    
    # Store the values in a dictionary
    entered_data = {
        "team1": team1,
        "team2": team2,
        "team1_country": team1_country,
        "team2_country": team2_country,
        "team1_group_rank": team1_group_rank,
        "team2_group_rank": team2_group_rank,
        "team1_group_win": team1_group_win,
        "team2_group_win": team2_group_win,
        "team1_group_loss": team1_group_loss,
        "team2_group_loss": team2_group_loss,
        "year": year,
        "tournament": tournament
    }
    
    # Display the dictionary
    st.write("Entered Data:")
    st.json(entered_data)

    col1, col2 = st.columns(2)
    #for model prediction
    
    inference_file_path = r'pickles/'
    trained_model,label_encoder,scaler,explainer = load_inference_artifacts(inference_file_path)
    trained_model_state_dict,label_encoder,scaler,explainer_ann = load_inference_artifacts_ann(inference_file_path)

    with col1:
        if st.button("Predict via XgBoost"):
            X_column_dict = read_X_train_cols()
            for_prediction_features = match_entry_data(entered_data, X_column_dict)
            prediction_results = predict(trained_model,label_encoder,scaler,explainer, for_prediction_features)
            explanation = prediction_results[0]['explanation']
            explanation_df = pd.DataFrame(explanation.items(), columns=['Feature', 'SHAP Value'])
            predicted_winner = entered_data['team1'] if prediction_results[0]['prediction'] == 1 else entered_data['team2']
            fig = px.bar(explanation_df, x='Feature', y='SHAP Value', title='Top 3 Features Contributing to Prediction')
            st.write(f"Predicted Winner: {predicted_winner}")
            st.plotly_chart(fig)

    with col2:
        if st.button("Predict via ANN"):
            X_column_dict = read_X_train_cols()
            for_prediction_features = match_entry_data(entered_data, X_column_dict)
            trained_model = ANNModel(len(X_column_dict))
            prediction_results = predict_ann(trained_model, trained_model_state_dict,label_encoder,scaler,explainer_ann, for_prediction_features)
            explanation = prediction_results[0]['explanation']
            explanation_df = pd.DataFrame(explanation.items(), columns=['Feature', 'SHAP Value'])
            predicted_winner = entered_data['team1'] if prediction_results[0]['prediction'] == 1 else entered_data['team2']
            fig = px.bar(explanation_df, x='Feature', y='SHAP Value', title='Top 3 Features Contributing to Prediction')
            st.write(f"Predicted Winner: {predicted_winner}")
            st.plotly_chart(fig)
        
def read_X_train_cols():
    df = pd.read_csv(r'files/mlbb/MSC/model_usage/X_train.csv')
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