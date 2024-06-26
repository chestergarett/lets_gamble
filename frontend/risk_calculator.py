import streamlit as st
from calculator.calculate import run_calculator_pipeline

def app():
    st.title("Risk Calculator")
    col1, col2 = st.columns(2)

    with col1:
        left_input = st.number_input("Left Column Input", value=0.0)

    with col2:
        if 'right_inputs' not in st.session_state:
            st.session_state.right_inputs = [0.0]

        zero_exists = False 
        for idx, val in enumerate(st.session_state.right_inputs):
            st.session_state.right_inputs[idx] = st.number_input(f"Right Column Input {idx+1}", value=val, key=f"right_input_{idx}", format="%.2f")
            if st.session_state.right_inputs[idx] == 0.0:
                zero_exists = True

        add_button_disabled = zero_exists

        if st.button("Add", disabled=add_button_disabled):
            st.session_state.right_inputs.append(0.0)
            st.experimental_rerun()

    if st.button('Calculate Risk'):
        capital = left_input
        odds_per_game = {index: item for index, item in enumerate(st.session_state.right_inputs)}
        allocation_per_game, possibilities = run_calculator_pipeline(capital,odds_per_game)

        st.write(f"If you allocate {allocation_per_game}, you have a chance to gain/lose the ff: {possibilities}")