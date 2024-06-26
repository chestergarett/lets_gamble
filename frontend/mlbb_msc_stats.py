import pandas as pd
import streamlit as st
import plotly.express as px

def app():
    csv = r'files/mlbb/MSC/all_years.csv'
    df = pd.read_csv(csv)
    columns_to_drop = ['Unnamed: 0', 'Unnamed: 0.1']
    df = df.drop(columns=columns_to_drop)
    df = df.drop_duplicates()
    df['Year'] = df['Year'].astype(str)

    grouped_df = df.groupby(['Year', 'winner_country', 'loser_country']).size().reset_index(name='count')
    st.title('MLBB MSC Stats')
    st.subheader("Country Head to Head")
    years = ["All Years"] + sorted(grouped_df['Year'].unique().tolist())
    selected_years = st.multiselect('Select Year(s)', years, default=["All Years"])

    winner_countries = sorted(grouped_df['winner_country'].unique())
    selected_winner_countries = st.multiselect('Select Winner Countries', winner_countries, winner_countries)

    loser_countries = sorted(grouped_df['loser_country'].unique())
    selected_loser_countries = st.multiselect('Select Loser Countries', loser_countries, loser_countries)


    if "All Years" in selected_years:
        filtered_df = grouped_df[
            (grouped_df['winner_country'].isin(selected_winner_countries)) &
            (grouped_df['loser_country'].isin(selected_loser_countries))
        ]
    else:
        filtered_df = grouped_df[
            (grouped_df['Year'].isin(selected_years)) &
            (grouped_df['winner_country'].isin(selected_winner_countries)) &
            (grouped_df['loser_country'].isin(selected_loser_countries))
        ]

    st.subheader(f"Per Country Head to Head for {selected_years}")
    st.dataframe(filtered_df)

    # Display filtered raw data
    if "All Years" in selected_years:
        filtered_raw_df = df[
            (df['winner_country'].isin(selected_winner_countries)) &
            (df['loser_country'].isin(selected_loser_countries))
        ][['winner', 'winner_country', 'loser', 'loser_country', 'winner_group_rank', 'loser_group_rank', 'Year']]
    else:
        filtered_raw_df = df[
            (df['Year'].isin(selected_years)) &
            (df['winner_country'].isin(selected_winner_countries)) &
            (df['loser_country'].isin(selected_loser_countries))
        ][['winner', 'winner_country', 'loser', 'loser_country', 'winner_group_rank', 'loser_group_rank', 'Year']]


    st.subheader("Per Country Head to Head Details")
    st.write(filtered_raw_df)

    ## group rankings
    grouped_rank_df = df.groupby(['Year', 'winner_group_rank', 'loser_group_rank']).size().reset_index(name='count')
    if "All Years" not in selected_years:
        grouped_rank_df = grouped_rank_df[grouped_rank_df['Year'].isin(selected_years)]

    less_rank_count = grouped_rank_df[grouped_rank_df['winner_group_rank'] > grouped_rank_df['loser_group_rank']]['count'].sum()
    greater_rank_count = grouped_rank_df[grouped_rank_df['winner_group_rank'] < grouped_rank_df['loser_group_rank']]['count'].sum()

    st.subheader("KPIs")
    col1, col2 = st.columns(2)
    col1.metric(label="Winner Group Rank < Loser Group Rank", value=less_rank_count)
    col2.metric(label="Winner Group Rank > Loser Group Rank", value=greater_rank_count)

    st.subheader(f"Group Rank Head to Head for {selected_years}")
    st.dataframe(grouped_rank_df)
