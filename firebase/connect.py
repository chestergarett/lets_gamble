import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd

def establish_connection():
    
    if not firebase_admin._apps:
        cred = credentials.Certificate('key.json')
        app = firebase_admin.initialize_app(cred)
    else:
        app = firebase_admin.get_app()

    return app

def get_bet_logs():
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    docs = bet_collection.stream()
    doc_array = []
    for doc in docs:
        doc_dict = doc.to_dict()
        doc_dict['id'] = doc.id
        doc_array.append(doc_dict)

    bet_df = pd.DataFrame(doc_array)
    bet_df.set_index('id', inplace=True)
    return bet_df

def get_model_prediction_logs():
    db = firestore.client()
    pred_collection = db.collection('model-hypothesis-test')
    docs = pred_collection.stream()
    doc_array = []
    for doc in docs:
        doc_dict = doc.to_dict()
        doc_dict['id'] = doc.id
        doc_array.append(doc_dict)

    pred_df = pd.DataFrame(doc_array)
    pred_df.set_index('id', inplace=True)
    return pred_df

def edit_single_bet(doc_id, updated_data):
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    bet_collection.document(doc_id).update(updated_data)

    print('Successfully edited record')

def edit_single_prediction(doc_id, updated_data):
    db = firestore.client()
    bet_collection = db.collection('model-hypothesis-test')
    bet_collection.document(doc_id).update(updated_data)

    print('Successfully edited record')

def add_bets_to_db(transactions):
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    for transaction in transactions:
        bet_collection.document().set(transaction)

    print('Transactions uploaded to Firestore')

def add_predicted_winners_to_db(transaction):
    db = firestore.client()
    predicted_winners_collection = db.collection('model-hypothesis-test')
    predicted_winners_collection.document().set(transaction)

    print('Transactions uploaded to Firestore')

def add_matches_to_db(transactions):
    db = firestore.client()
    match_odds_collection = db.collection('match-odds-log')
    for transaction in transactions:
        match_odds_collection.document().set(transaction)

    print('Transactions uploaded to Firestore')

def run_firebase_pipeline():
    establish_connection()
    bet_df = get_bet_logs()
    
    return bet_df

def run_save_match_odds(transactions):
    establish_connection()
    add_matches_to_db(transactions)
    
def run_save_transactions_pipeline(transactions):
    establish_connection()
    add_bets_to_db(transactions)

def run_save_predicted_winners(transaction):
    establish_connection()
    add_predicted_winners_to_db(transaction)

def run_ab_test_pipeline():
    establish_connection()
    pred_df = get_model_prediction_logs()

    return pred_df