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

def get_match_odds_logs():
    db = firestore.client()
    odds_collection = db.collection('match-odds-log')
    docs = odds_collection.stream()
    doc_array = []
    for doc in docs:
        doc_dict = doc.to_dict()
        doc_dict['id'] = doc.id
        doc_array.append(doc_dict)

    odds_df = pd.DataFrame(doc_array)
    odds_df.set_index('id', inplace=True)
    return odds_df

def edit_single_bet(doc_id, updated_data):
    establish_connection()
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    bet_collection.document(doc_id).update(updated_data)

    print('Successfully edited record')

def edit_single_match(doc_id, updated_data):
    establish_connection()
    db = firestore.client()
    match_collection = db.collection('match-odds-log')
    match_collection.document(doc_id).update(updated_data)

    print('Successfully edited record')

def read_single_match(doc_id):
    establish_connection()
    db = firestore.client()
    match_log_ref = db.collection('match-odds-log').document(doc_id)
    match_log = match_log_ref.get()
    match_log = match_log.to_dict()
    
    if match_log['winner']=='':
        return True
    return False

def edit_single_prediction(doc_id, updated_data):
    establish_connection()
    db = firestore.client()
    bet_collection = db.collection('model-hypothesis-test')
    bet_collection.document(doc_id).update(updated_data)

    print('Successfully edited record')

def add_bets_to_db(transactions):
    db = firestore.client()
    bet_collection = db.collection('bet-log')
    for transaction in transactions:
        bet_collection.document().set(transaction)

    if len(transactions)>0:
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
        right_team_name = transaction['right_team_name']
        right_odds = transaction['right_odds']
        left_team_name = transaction['left_team_name']
        left_odds = transaction['left_odds']
        draw_odds = transaction['draw_odds']
        date = transaction['date']
        game = transaction['game']
        existing_transactions = match_odds_collection.where('right_team_name', '==', right_team_name) \
                                                     .where('right_odds', '==', right_odds) \
                                                     .where('left_team_name', '==', left_team_name) \
                                                     .where('left_odds', '==', left_odds) \
                                                     .where('draw_odds', '==', draw_odds) \
                                                     .where('date', '==', date) \
                                                     .where('game', '==', game) \
                                                     .stream()
        

        if any(existing_transactions):
            print(f"Transaction already exists: {transaction}")
        else:
            match_odds_collection.document().set(transaction)
            print(f'Transactions uploaded to Firestore: {transaction}')

def query_blank_winners():
    establish_connection()
    db = firestore.client()
    collection_ref = db.collection('model-hypothesis-test')
    query = collection_ref.where('actual_winner', '==', '')
    docs = query.stream()

    return docs

def delete_odds_documents(collection):
    establish_connection()
    db = firestore.client()
    collection_ref = db.collection(collection)
    query = collection_ref.where('date', '==', None)
    docs = query.stream()
    for doc in docs:
        doc.reference.delete()  # Delete the document
        print(f'Deleted document with ID: {doc.id}')

def check_if_update_needed(doc_id,row):
    establish_connection()
    db = firestore.client()
    collection_ref = db.collection('bet-log')
    doc = collection_ref.document(doc_id)
    doc_dict = doc.get().to_dict()
    if ('win_loss_code' not in doc_dict) or ('win_loss_amount' not in doc_dict) or (doc_dict['win_loss_code']!=row['win_loss_code']) or (doc_dict['win_loss_amount'] != row['win_loss_amount']) or (doc_dict['tournament'] != row['tournament']) or (doc_dict['game'] != row['game']) or (doc_dict['bet_date'] != row['bet_date']):
        update = {'win_loss_code': row['win_loss_code'], 
                  'win_loss_amount': row['win_loss_amount'], 
                  'tournament': row['tournament'], 
                  'bet_date': row['bet_date'],
                  'game': row['game']}
        
        print(update)
        doc.update(update)

        print(f'Updated Bet record {doc_id}')
        
def run_firebase_pipeline(data_type):
    establish_connection()
    if data_type=='bet_logs':
        bet_df = get_bet_logs()
        return bet_df
    if data_type=='match_logs':
        match_odds_df = get_match_odds_logs()
        return match_odds_df
    if data_type=='model_prediction_logs':
        pred_df = get_model_prediction_logs()
        return pred_df
    
    return 'Unknown document collection'

def run_save_match_odds(transactions):
    establish_connection()
    add_matches_to_db(transactions)
    
def run_save_transactions_pipeline(transactions):
    establish_connection()
    add_bets_to_db(transactions)

def run_save_predicted_winners(transaction):
    establish_connection()
    add_predicted_winners_to_db(transaction)
