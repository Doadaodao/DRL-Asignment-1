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

    
    station_north = int((taxi_row-1, taxi_col) in stations)
    station_south = int((taxi_row+1, taxi_col) in stations)
    station_east = int((taxi_row, taxi_col+1) in stations)
    station_west = int((taxi_row, taxi_col-1) in stations)

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

    # features.append(station_north)
    # features.append(station_south)
    # features.append(station_east)
    # features.append(station_west)

    # features.append(passenger_look)
    # features.append(destination_look)

    # features.append(at_passenger)
    # features.append(at_destination)
    
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


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # -------------------------------
# # Feature Extraction Function
# # -------------------------------
# def extract_features(obs):
#     """
#     Extract features from an observation in the custom taxi environment.

#     Parameters:
#     - obs: A tuple with 16 elements containing:
#          (taxi_row, taxi_col,
#           station0_row, station0_col, station1_row, station1_col,
#           station2_row, station2_col, station3_row, station3_col,
#           obstacle_north, obstacle_south, obstacle_east, obstacle_west,
#           passenger_look, destination_look)
#     - grid_size: The size (n) of the grid.

#     Returns:
#     - A NumPy array of extracted features.
#     """
#     taxi_row, taxi_col = obs[0], obs[1]
#     stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
#     obstacle_flags = list(obs[10:14])
#     passenger_flag = obs[14]
#     destination_flag = obs[15]
    
#     taxi_row_norm = taxi_row
#     taxi_col_norm = taxi_col
#     features = [taxi_row_norm, taxi_col_norm]
    
#     for (s_row, s_col) in stations:
#         diff_row = (s_row - taxi_row)
#         diff_col = (s_col - taxi_col)
#         manhattan_dist = (abs(s_row - taxi_row) + abs(s_col - taxi_col))
#         features.extend([diff_row, diff_col, manhattan_dist])
    
#     features.extend(obstacle_flags)
#     features.append(passenger_flag)
#     features.append(destination_flag)
    
#     return np.array(features)

# # -------------------------------
# # Policy Network Definition
# # -------------------------------
# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim=20, hidden_dim=64, output_dim=6):
#         """
#         A simple MLP policy network.
#         """
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         # Return softmax probabilities for actions.
#         return F.softmax(x, dim=-1)

# # -------------------------------
# # Global Model Loading
# # -------------------------------
# # Use GPU if available.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# policy_model = None

# def load_model():
#     """
#     Loads the trained model from disk if it is not already loaded.
#     """
#     global policy_model
#     if policy_model is None:
#         policy_model = PolicyNetwork().to(device)
#         # Load the state dictionary. Adjust the path if needed.
#         policy_model.load_state_dict(torch.load("policy_model_training_env_lre4.pkl", map_location=device))
#         policy_model.eval()  # Set the model to evaluation mode.

# # -------------------------------
# # get_action Function for Testing
# # -------------------------------
# def get_action(obs):
#     """
#     Returns an action based on the given observation using the trained policy network.
    
#     Parameters:
#         obs (tuple): The observation from the environment.
    
#     Returns:
#         int: An action in the range 0-5.
#     """
#     # Load the model if it hasn't been loaded already.
#     load_model()
    
#     # Determine the grid_size. Here we assume a default value (e.g., 5).
#     # If your environment passes the grid size or it is encoded in the observation,
#     # update this accordingly.
#     grid_size = 5  
    
#     # Extract features using the same function as in training.
#     features = extract_features(obs)
#     features_tensor = torch.from_numpy(features).float().to(device)
    
#     # Forward pass: obtain action probabilities.
#     with torch.no_grad():
#         action_probs = policy_model(features_tensor)
    
#     # Choose the action with the highest probability (deterministic selection for testing).
#     action = torch.argmax(action_probs).item()
#     return action
