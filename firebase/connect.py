import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd

def establish_connection():
    cred = credentials.Certificate('key.json')
    app = firebase_admin.get_app()
    if not app:
        app = firebase_admin.initialize_app(cred)

    return app

def get_bet_logs():
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    docs = bet_collection.stream()
    doc_array = []
    for doc in docs:
        doc_array.append(doc.to_dict())

    bet_df = pd.DataFrame(doc_array)
    return bet_df

def add_bets_to_db(transactions):
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    for transaction in transactions:
        bet_collection.document().set(transaction)

    print('Transactions uploaded to Firestore')

def run_firebase_pipeline():
    establish_connection()
    bet_df = get_bet_logs()
    
    return bet_df

def run_save_transactions_pipeline(transactions):
    establish_connection()
    add_bets_to_db(transactions)
# run_firebase_pipeline()