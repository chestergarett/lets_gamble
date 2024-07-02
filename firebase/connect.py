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

def edit_single_bet(doc_id, updated_data):
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    bet_collection.document(doc_id).update(updated_data)

    print('Successfully edited record')

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