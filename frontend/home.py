import streamlit as st

profit = '27.64%'

profit_per_game = {   
    'mlbb': '42.46%',
    'hok':'19.17%',
    'valorant': '0.1%'
}

matches = 62
def app():
    st.title('Holy Grail of Sports Betting')
    st.write('Abstract')
    st.write(f'''In traditional financial markets, retail investors face significant challenges: 1) identifying optimal times to BUY, HOLD, and SELL, 2) competing against market whales, and 3) managing biases caused by human emotions. For sports betting, although human emotions still heavily influence decisions however the time variability is negligible and market whales are fewer. This system exploits these factors. To date, the devised system has generated a profit of {profit} from {matches} bets.
     ''')
    
    st.write('Hypothesis')
    st.write(f'''
        The system hypothesizes that betting against popular public bets—referred to as "fading the public" in betting terminology—will yield profitable outcomes.
    ''')
    
    st.write('Assumptions & Constraints')
    st.write(f'''
        1. Risk Management: An equal bet amount will be placed on each match until a desired profit level is achieved. Subsequently, bet amounts will be gradually increased for the next set of bets.
        2. Focus on E-sports: E-sports are chosen due to:
            a. The absence of physical attributes (e.g., height, strength), allowing focus on mental attributes.
            b. The dynamic nature of the meta, with game developers frequently updating game aspects, leading to frequent shifts in current powerhouses and laggards.
        3. Game-Specific Assessments: Each game will be evaluated differently, with various subsystems employing different approaches.
    ''')
    
    st.write('Methodology')
    st.write(f'''
        The general strategy is to fade the public. However, deviations from this general strategy are employed for different games:
            Mobile Legends: Bang Bang (MLBB): Fade the public during group/elimination stages; use AI model predictions during playoffs.
            Honor of Kings (HOK): Fade the public during group/elimination stages.
            Valorant: Fade the public during group/elimination stages.
            League of Legends (LOL): Fade the public during group/elimination stages.
    ''')

    st.write('Results & Findings')
    st.write(f'''
       Turns out, the public often get it right. As such, we need to spot what they believe is wrong and we bet with that team. After, post bet-analysis and match simulations current profit stands at {profit}.
        Breakdown per game are as follows:
    ''')
    for key, value in profit_per_game.items():
        st.write(f'{key} with profit of {value}')

    st.write('''
       The best bet performance is observed in MLBB, attributed to the author's in-depth knowledge of the game and the incorporation of qualitative factors and simulations into decision-making models. For other games, bets are placed primarily based on odds. To enhance yields without detailed game knowledge, an odds simulation model was developed, identifying bets with odds of 2 and 3 as yielding the best results.
    ''')

    st.write('Conclusion')
    st.write(f'''The findings indicate consistent profitability in the system. This research underscores the importance of strategic betting against public sentiment and the value of game-specific knowledge and simulation models in maximizing betting profits.''')
    return