from itertools import combinations

def allocate_capital_per_game(capital, odds_per_game):
    num_games = len(odds_per_game.keys())
    allocation = int((capital / num_games))
    allocation_per_game = {key: allocation for key in odds_per_game.keys()}
    return allocation_per_game

def compute_possibilities(allocation_per_game, odds_per_game):
    wins_losses = []
    possibilities = []
    num_games = len(odds_per_game.keys())

    for key, allocation in allocation_per_game.items():
        odds = odds_per_game[key]
        possible_win = (odds * allocation)
        possible_loss = allocation * -1
        wins_losses.append(possible_win)
        wins_losses.append(possible_loss)
    
    win_loss_combinations = list(combinations(wins_losses,num_games))

    for permutation in win_loss_combinations:
        possibility = sum(permutation)
        possibilities.append(possibility)
    
    return sorted(set(possibilities))

def run_calculator_pipeline(capital,odds_per_game):
    allocation_per_game = allocate_capital_per_game(capital, odds_per_game)
    possibilities = compute_possibilities(allocation_per_game, odds_per_game)

    return allocation_per_game,possibilities


# odds_per_game = {1: 2, 2: 1.5, 3: 2.4, 4: 1.5}
# capital = 10000

# allocation_per_game = allocate_capital_per_game(capital, odds_per_game)
# possibilities = compute_possibilities(allocation_per_game, odds_per_game)
# print(possibilities)
