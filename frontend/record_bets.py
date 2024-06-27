import streamlit as st
from firebase.connect import run_save_transactions_pipeline

def app():
    # Initialize session state to store transactions and count of rows
    if 'transactions' not in st.session_state:
        st.session_state['transactions'] = []
    if 'transaction_count' not in st.session_state:
        st.session_state['transaction_count'] = 1

    # Function to add a new row of input fields
    def add_transaction():
        # Check if all fields are filled
        for i in range(st.session_state['transaction_count']):
            if (st.session_state.get(f'game_{i}', '') == '' or
                st.session_state.get(f'tournament_{i}', '') == '' or
                st.session_state.get(f'bet_with_{i}', '') == '' or
                st.session_state.get(f'bet_against_{i}', '') == '' or
                st.session_state.get(f'odds_{i}', 0) == 0 or
                st.session_state.get(f'bet_amount_{i}', 0) == 0):
                st.error('Fields not filled')
                return

        # Increment the transaction count
        st.session_state['transaction_count'] += 1
        st.experimental_rerun()

    # Function to save all transactions
    def save_transactions():
        # Clear existing transactions before saving
        st.session_state['transactions'] = []
        for i in range(st.session_state['transaction_count']):
            transaction = {
                'game': st.session_state[f'game_{i}'],
                'tournament': st.session_state[f'tournament_{i}'],
                'odds': st.session_state[f'odds_{i}'],
                'bet_amount': st.session_state[f'bet_amount_{i}'],
                'bet_with': st.session_state[f'bet_with_{i}'],
                'bet_against': st.session_state[f'bet_against_{i}']
            }
            st.session_state['transactions'].append(transaction)

        run_save_transactions_pipeline(st.session_state['transactions'])
        st.write("Transactions Saved!")
        

    # Display input fields dynamically
    for i in range(st.session_state['transaction_count']):
        cols = st.columns(6)
        with cols[0]:
            st.text_input('Game', key=f'game_{i}')
        with cols[1]:
            st.text_input('Tournament', key=f'tournament_{i}')
        with cols[2]:
            st.number_input('Odds', key=f'odds_{i}', step=0.01, format="%.2f")
        with cols[3]:
            st.number_input('Bet Amount', key=f'bet_amount_{i}', step=1.0, format="%.2f")
        with cols[4]:
            st.text_input('Bet With', key=f'bet_with_{i}')
        with cols[5]:
            st.text_input('Bet Against', key=f'bet_against_{i}')

    # Add transaction button
    if st.button('Add'):
        add_transaction()

    # Save transactions button
    if st.button('Save'):
        save_transactions()

    # Display the list of transactions
    if st.session_state['transactions']:
        st.write("Transactions:")
        for transaction in st.session_state['transactions']:
            st.write(transaction)