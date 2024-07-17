import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firebase.connect import delete_odds_documents



delete_odds_documents('match-odds-log')