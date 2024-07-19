import pandas as pd
import streamlit as st
import plotly.express as px
import os
from dotenv import load_dotenv
from jobs.get_all_firebase_data import load_offline_df
from firebase.connect import edit_single_prediction

load_dotenv()
AUTHENTICATION_KEY = os.environ.get('AUTHENTICATION_KEY')

def app():
    pred_df = load_offline_df('model_prediction_logs')
    pred_df = pred_df.reset_index()
    ordered_columns = ['model', 'team1', 'team2', 'predicted_winner', 'actual_winner', 'tournament']
    pred_df = pred_df[ordered_columns]
    pred_df['correct'] = pred_df['predicted_winner'] == pred_df['actual_winner']
    accuracy_df = pred_df.groupby('model')['correct'].mean().reset_index()
    accuracy_df.columns = ['model', 'accuracy']
    print(accuracy_df[accuracy_df['model']=='XgBoost']['accuracy'])
    xg_boost_accuracy = accuracy_df[accuracy_df['model']=='XgBoost']['accuracy'].values[0]
    ann_accuracy = accuracy_df[accuracy_df['model']=='ANN']['accuracy'].values[0]
    st.title("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:    
        st.markdown(f"<div style='font-weight: bold'>XgBoost Accuracy: </div> <div style='text-align: center; color:green; font-size: 24px'>{xg_boost_accuracy:.2%}</div>", unsafe_allow_html=True)
    with col2:    
        st.markdown(f"<div style='font-weight: bold'>ANN Accuracy: </div> <div style='text-align: center; color:green; font-size: 24px'>{ann_accuracy:.2%}</div>", unsafe_allow_html=True)

    st.title("Model A/B Test Details")
    st.dataframe(pred_df)

                
