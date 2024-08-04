import streamlit as st
import pandas as pd
import os
from models.mlbb_msc_xgboost import predict, load_inference_artifacts
from models.mlbb_msc_ann import predict_ann, load_inference_artifacts_ann
from firebase.connect import run_save_predicted_winners
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
    st.title('MLBB Predict Winner')

    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
        st.session_state.entered_data = {}
        st.session_state.authenticated = False
        st.session_state.prediction_model = None

    # Create columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Team 1")
        team1 = st.text_input("Team 1", key="team1")
        team1_country = st.text_input("Team 1 Country", key="team1_country")
        team1_group_rank = st.number_input("Team 1 Group Rank", min_value=1, step=1, key="team1_group_rank")
        team1_group_win = st.number_input("Team 1 Group Wins", min_value=0, step=1, key="team1_group_win")
        team1_group_loss = st.number_input("Team 1 Group Losses", min_value=0, step=1, key="team1_group_loss")

    with col2:
        st.header("Team 2")
        team2 = st.text_input("Team 2", key="team2")
        team2_country = st.text_input("Team 2 Country", key="team2_country")
        team2_group_rank = st.number_input("Team 2 Group Rank", min_value=1, step=1, key="team2_group_rank")
        team2_group_win = st.number_input("Team 2 Group Wins", min_value=0, step=1, key="team2_group_win")
        team2_group_loss = st.number_input("Team 2 Group Losses", min_value=0, step=1, key="team2_group_loss")

    # Year and Tournament fields spanning both columns
    year = st.number_input("Year", min_value=2017, max_value=2024, step=1, key="year")
    tournament = st.selectbox("Tournament", options=["MSC", "Other"], key="tournament")

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
    # for model prediction

    inference_file_path = r'pickles/'
    trained_model, label_encoder, scaler, explainer = load_inference_artifacts(inference_file_path)
    trained_model_state_dict, label_encoder, scaler, explainer_ann = load_inference_artifacts_ann(inference_file_path)

    with col1:
        if st.button("Predict via XgBoost"):
            st.session_state.prediction_made = True
            st.session_state.entered_data = entered_data
            st.session_state.prediction_model = 'XgBoost'
            X_column_dict = read_X_train_cols()
            for_prediction_features = match_entry_data(entered_data, X_column_dict)
            prediction_results = predict(trained_model, label_encoder, scaler, explainer, for_prediction_features)
            explanation = prediction_results[0]['explanation']
            explanation_df = pd.DataFrame(explanation.items(), columns=['Feature', 'SHAP Value'])
            st.session_state.predicted_winner = entered_data['team1'] if prediction_results[0]['prediction'] == 1 else entered_data['team2']
            fig = px.bar(explanation_df, x='Feature', y='SHAP Value', title='Top 3 Features Contributing to Prediction')
            st.write(f"Predicted Winner: {st.session_state.predicted_winner}")
            st.plotly_chart(fig)

        if st.session_state.prediction_made and st.session_state.prediction_model == 'XgBoost':
            st.session_state.entered_data['model'] = 'XgBoost'
            st.session_state.entered_data['predicted_winner'] = st.session_state.predicted_winner
            st.session_state.entered_data['actual_winner'] = ''

            auth_key = st.text_input("Enter Authentication Key", type="password", key="auth_key_xgboost")
            if auth_key == os.environ['AUTHENTICATION_KEY']:
                st.session_state.authenticated = True
                st.success("Authentication successful.")
            else:
                st.session_state.authenticated = False

            if st.session_state.authenticated:
                if st.button("Save XgBoost Prediction"):
                    try:
                        run_save_predicted_winners(st.session_state.entered_data)
                        st.success('Prediction saved successfully.')
                    except Exception as e:
                        st.error(f"Error saving prediction: {e}")
                        print(f"Error: {e}")
            else:
                st.warning("Please enter a valid authentication key to save the prediction.")

    with col2:
        if st.button("Predict via ANN"):
            st.session_state.prediction_made = True
            st.session_state.entered_data = entered_data
            st.session_state.prediction_model = 'ANN'
            X_column_dict = read_X_train_cols()
            for_prediction_features = match_entry_data(entered_data, X_column_dict)
            trained_model = ANNModel(len(X_column_dict))
            prediction_results = predict_ann(trained_model, trained_model_state_dict, label_encoder, scaler, explainer_ann, for_prediction_features)
            explanation = prediction_results[0]['explanation']
            explanation_df = pd.DataFrame(explanation.items(), columns=['Feature', 'SHAP Value'])
            st.session_state.predicted_winner = entered_data['team1'] if prediction_results[0]['prediction'] == 1 else entered_data['team2']
            fig = px.bar(explanation_df, x='Feature', y='SHAP Value', title='Top 3 Features Contributing to Prediction')
            st.write(f"Predicted Winner: {st.session_state.predicted_winner}")
            st.plotly_chart(fig)

        if st.session_state.prediction_made and st.session_state.prediction_model == 'ANN':
            st.session_state.entered_data['model'] = 'ANN'
            st.session_state.entered_data['predicted_winner'] = st.session_state.predicted_winner
            st.session_state.entered_data['actual_winner'] = ''

            auth_key = st.text_input("Enter Authentication Key", type="password", key="auth_key_ann")
            if auth_key == os.environ['AUTHENTICATION_KEY']:
                st.session_state.authenticated = True
                st.success("Authentication successful.")
            else:
                st.session_state.authenticated = False

            if st.session_state.authenticated:
                if st.button("Save ANN Prediction"):
                    try:
                        run_save_predicted_winners(st.session_state.entered_data)
                        st.success('Prediction saved successfully.')
                    except Exception as e:
                        st.error(f"Error saving prediction: {e}")
                        print(f"Error: {e}")
            else:
                st.warning("Please enter a valid authentication key to save the prediction.")
        
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