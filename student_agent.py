import pickle
import random
import numpy as np
import os
import torch
import torch.nn as nn

# Define the DQN network (must match the training architecture)
# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_dim)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # Global constants (update these if your state dimension is different)
# STATE_DIM = 16      # e.g., length of the state tuple (this should match your environment)
# ACTION_DIM = 6      # six possible discrete actions

# # Load the trained model once (this code is executed when the module is imported)
# MODEL_PATH = "dqn_taxi_model.pkl"
# if os.path.exists(MODEL_PATH):
#     trained_model = DQN(STATE_DIM, ACTION_DIM)
#     trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
#     trained_model.eval()  # set the model to evaluation mode
# else:
#     trained_model = None
#     print(f"Warning: Model file {MODEL_PATH} not found. get_action will not work properly.")


# Feature extractor to convert raw state to feature vector
def extract_features(state):
    taxi_row, taxi_col, s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = state

    features = []
    
    # Create station list
    stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]

    # for station in stations:
    #     features.append(station[0] - taxi_row)
    #     features.append(station[1] - taxi_col)
    
    features.append(obstacle_north)
    features.append(obstacle_south)
    features.append(obstacle_east)
    features.append(obstacle_west)

    features.append(passenger_look)
    features.append(destination_look)
    
    return tuple(features)



# Load Q-table once, at import time
# Make sure "q_table.pkl" is in the same directory or provide correct path
q_table = {}
if os.path.exists("q_table_at_feature_2_9999_10000.pkl"):
    with open("q_table_at_feature_2_9999_10000.pkl","rb") as f:
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
    
    if np.random.rand() < 0.0001:
        action = np.random.choice([0, 1, 2, 3, 4, 5])
    else:
        q_values = get_q_values(feature)
        action = int(np.argmax(q_values))
    # q_values = get_q_values(obs)
    # action = int(np.argmax(q_values))
    # return random.choice([0, 1, 2, 3, 4, 5])

    # if trained_model is None:
    #     # If the model isn't loaded, you could either raise an error or return a random action.
    #     raise ValueError("Trained model not loaded. Please ensure that dqn_taxi_model.pkl exists.")
    
    # # Convert observation (tuple) to a PyTorch tensor.
    # # Ensure the input has the correct shape: [batch_size, STATE_DIM]
    # obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0)  # shape: (1, STATE_DIM)
    
    # # Forward pass through the network to get Q-values for each action.
    # with torch.no_grad():
    #     q_values = trained_model(obs_tensor)
    
    # # Choose the action with the highest Q-value.
    # action = q_values.argmax(dim=1).item()



    # action = np.random.choice([0, 1, 2, 3, 4, 5])
    return action
