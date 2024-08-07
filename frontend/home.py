import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jobs.get_all_firebase_data import load_offline_df


def app():
    bet_df = load_offline_df('bet_logs')
    bet_df['win_loss_amount'] = bet_df['win_loss_amount'].astype(float)
    bet_df['bet_amount'] = bet_df['bet_amount'].astype(float)
    total_wins = bet_df[bet_df['win_loss_code'] == 'WIN']['win_loss_amount'].sum()
    total_bet_amounts = bet_df[bet_df['win_loss_code'].isin(['WIN', 'LOSS'])]['bet_amount'].sum()
    total_matches = len(bet_df[bet_df['win_loss_code'].isin(['WIN', 'LOSS'])])
    profit = (total_wins - total_bet_amounts) / total_bet_amounts

    profit_per_game = {}

    for game in bet_df['game'].unique():
        game_df = bet_df[bet_df['game'] == game]
        game_wins = game_df[game_df['win_loss_code'] == 'WIN']['win_loss_amount'].sum()
        game_bet_amounts = game_df[game_df['win_loss_code'].isin(['WIN', 'LOSS'])]['bet_amount'].sum()
        if game_bet_amounts > 0:
            game_profit = ((game_wins-game_bet_amounts) / game_bet_amounts)
            profit_per_game[game] = f'{game_profit * 100:.2f}%'

    st.markdown(
        """
        <style>
        .subheader {
            font-size: 1.5em;
            font-weight: bold;
        }
        .highlight {
            color: green;
            font-weight: bold;
        }
        .game-name {
            font-weight: bold;
        }
        .justified-text {
            text-align: justify;
            margin: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title('Holy Grail of Sports Betting')
    st.write('### Authored by: Chester Garett Calingacion')
    st.write('### Page last updated on: Aug 4, 2024')
    st.write('### **Introduction**')
    st.write(f'''<div class='justified-text'>
             In traditional financial markets, retail investors face significant challenges: 
                1) identifying optimal times to BUY, HOLD, and SELL, 
                2) competing against market whales, and
                 3) managing biases caused by human emotions. 
                For sports betting, although human emotions still heavily influence decisions - the time variability is negligible and market whales are fewer. 
                This system exploits these factors. To date, the devised system has generated 
                a profit of <span class='highlight'>{profit: .2%}</span> from <span class='highlight'>{total_matches}</span> bets.
            </div>
     ''',unsafe_allow_html=True)
    
    st.write('### **Hypothesis**')
    st.write(f'''
        <div class='justified-text'>
        The system hypothesizes that betting against popular public bets—referred to as "fading the public" in betting terminology—will yield profitable outcomes.
        </div>
    ''',unsafe_allow_html=True)
    
    st.write('### **Assumptions & Constraints**')
    st.write(f'''
        <div class='justified-text'>
            <ul>     
                <li><strong> 1. Risk Management </strong>: An equal bet amount will be placed on each match until a desired profit level is achieved. Subsequently, bet amounts will be gradually increased for the next set of bets. </li>
                <li><strong> 2. Focus on E-sports</strong>: E-sports are chosen due to:
                    <ul>
                        <li><em>a</em>. The absence of physical attributes (e.g., height, strength), allowing focus on mental attributes. </li>
                        <li><em>b</em>. The dynamic nature of the meta, with game developers frequently updating game aspects, leading to frequent shifts in current powerhouses and laggards. </li>
                    </ul>
                </li>
               <li><strong>3. Game-Specific Assessments</strong>: Each game will be evaluated differently, with various subsystems employing different approaches.</li>
            </ul>
        </div>
    ''',unsafe_allow_html=True)
    
    st.write('### **Methodology**')
    st.write('''
        <div class='justified-text'>
        The general strategy is to fade the public. However, deviations from this general strategy are employed for different games:
             <ul>     
                <li><span class='game-name'>Mobile Legends: Bang Bang (MLBB)</span>: Fade the public during group/elimination stages; use AI model predictions during playoffs. </li>
                <li><span class='game-name'>Honor of Kings (HOK)</span>: Fade the public during group/elimination stages.</li>
                <li><span class='game-name'>Valorant</span>: Fade the public during group/elimination stages.</li>
                <li><span class='game-name'>League of Legends (LOL)</span>: Fade the public during group/elimination stages.</li>
             </ul>
        </div>
    ''', unsafe_allow_html=True)

    st.write('### **Results & Findings**')
    st.write(f'''
        <div class='justified-text'>
            Turns out, the public sometimes get it right. 
            As such, being selective when we bet against the public and further filter down prospective bets will maximize profit results. 
            After, post bet-analysis and match simulations current profit stands at <span class='highlight'>{profit:.2%}</span>.
            Breakdown per game are as follows:
        </div>
        ''', unsafe_allow_html=True)
    
    for key, value in profit_per_game.items():
        st.write(f'<div class="justified-text"><span class="game-name">{key}</span> with profit of <span class="highlight">{value}</span></div>', unsafe_allow_html=True)

    st.write('''
        <div class='justified-text'>
        The best bet performance is observed in <span class="game-name">MLBB</span>, attributed to the author's in-depth knowledge of the game and the incorporation of qualitative factors 
        and simulations into decision-making models. 
        For other games, bets are placed primarily based on odds. 
             To enhance yields without detailed game knowledge, an odds simulation model was developed, 
             identifying bets with odds of <span class='highlight'>2 and 3</span> as yielding the best results.
        </div>
    ''',unsafe_allow_html=True)

    st.write('### **Conclusion**')
    st.write(f'''
             <div class='justified-text'>
             The findings indicate consistent profitability in the system. 
             This research underscores the importance of strategic betting against public sentiment and the value of game-specific knowledge and simulation models 
             in maximizing betting profits.
             </div>
             ''',unsafe_allow_html=True)
    