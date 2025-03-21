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
    
    # Create station list
    stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]

    at_passenger = 0
    for station in stations:
        if taxi_row == station[0] and taxi_col == station[1] and passenger_look:
            at_passenger = 1

    at_destination = 0
    for station in stations:
        if taxi_row == station[0] and taxi_col == station[1] and destination_look:
            at_destination = 1

    # for station in stations:
    #     features.append(station[0] - taxi_row)
    #     features.append(station[1] - taxi_col)
    
    features.append(obstacle_north)
    features.append(obstacle_south)
    features.append(obstacle_east)
    features.append(obstacle_west)

    features.append(passenger_look)
    features.append(destination_look)

    # features.append(at_passenger)
    # features.append(at_destination)
    
    return tuple(features)

# Load Q-table once, at import time
# Make sure "q_table.pkl" is in the same directory or provide correct path
q_table = {}
if os.path.exists("q_table_2_9999_30000.pkl"):
    with open("q_table_2_9999_30000.pkl","rb") as f:
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

    # if trained_model is None:
    #     # If the model isn't loaded, you could either raise an error or return a random action.
    #     raise ValueError("Trained model not loaded. Please ensure that dqn_taxi_model.pkl exists.")
    
    # # Convert observation (tuple) to a PyTorch tensor.
    # # Ensure the input has the correct shape: [batch_size, STATE_DIM]
    # obs_tensor = torch.FloatTensor(np.array(feature)).unsqueeze(0)  # shape: (1, STATE_DIM)
    
    # # Forward pass through the network to get Q-values for each action.
    # with torch.no_grad():
    #     q_values = trained_model(obs_tensor)
    
    # # Choose the action with the highest Q-value.
    # action = q_values.argmax(dim=1).item()



    # action = np.random.choice([0, 1, 2, 3, 4, 5])
    return action
