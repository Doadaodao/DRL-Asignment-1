import pickle
import random
import numpy as np
import os
import torch
import torch.nn as nn


# Feature extractor to convert raw state to feature vector
def extract_features(state):
    taxi_row, taxi_col, s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = state

    features = []
    
    features.append(obstacle_north)
    features.append(obstacle_south)
    features.append(obstacle_east)
    features.append(obstacle_west)

    return tuple(features)

# Load Q-table once, at import time
# Make sure "q_table.pkl" is in the same directory or provide correct path
q_table = {}
if os.path.exists("q_table_avoid_obstacle.pkl"):
    with open("q_table_avoid_obstacle.pkl","rb") as f:
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

    feature = extract_features(obs)
    
    
    q_values = get_q_values(feature)
    action = int(np.argmax(q_values))


    # action = np.random.choice([0, 1, 2, 3, 4, 5])
    return action