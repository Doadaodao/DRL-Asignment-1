import pickle
import random
import numpy as np
import os

# Load Q-table once, at import time
# Make sure "q_table.pkl" is in the same directory or provide correct path
q_table = {}
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl","rb") as f:
        q_table = pickle.load(f)

def get_q_values(state):
    # Fallback: if state is unknown, return all 0.0 or random
    if state not in q_table:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return q_table[state]

def get_action(obs):
    """
    obs is the environment state (tuple).
    Return an integer from 0-5 that picks the best known action.
    """
    q_values = get_q_values(obs)
    action = int(np.argmax(q_values))
    # return random.choice([0, 1, 2, 3, 4, 5])
    return action
