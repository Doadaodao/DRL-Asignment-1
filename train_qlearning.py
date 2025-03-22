import random
import pickle
import numpy as np

from training_taxi_environment import TrainingTaxiEnv
from custom_taxi_env import CustomTaxiEnv
from simple_custom_taxi_env import SimpleTaxiEnv

# Hyperparameters
alpha = 0.2          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.9999  # Epsilon decays each step or episode

num_episodes = 30000  # Increase as needed
max_steps_per_episode = 4000  # Just a safeguard if you want

# Q-Table as a dictionary: {state: [Q-values for each action]}
q_table = {}

def get_q_values(state):
    """Return the Q-values for a given state, initializing if necessary."""
    if state not in q_table:
        # 6 possible actions: [0,1,2,3,4,5]
        q_table[state] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return q_table[state]

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

    features.append(station_north)
    features.append(station_south)
    features.append(station_east)
    features.append(station_west)

    features.append(passenger_look)
    features.append(destination_look)

    # features.append(at_passenger)
    # features.append(at_destination)
    
    return tuple(features)

env = TrainingTaxiEnv(min_size = 6, max_size=8)

rewards_per_episode = []

for episode in range(num_episodes):
    state, _ = env.reset()
    feature = extract_features(state)
    total_reward = 0

    done = False

    while not done:
        # Initialize the state in the Q-table if it is not already present.
        if feature not in q_table:
            q_table[feature] = np.zeros(6)

        # Implement an ε-greedy policy for action selection.
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(q_table[feature])

        next_state, reward, done, _ = env.step(action)
        next_feature = extract_features(state)
        total_reward += reward

        if next_feature not in q_table:
            q_table[next_feature] = np.zeros(6)
        
        best_next_action = np.argmax(q_table[next_feature])
        td_target = reward + gamma * q_table[next_feature][best_next_action]
        td_error = td_target - q_table[feature][action]
        q_table[feature][action] += alpha * td_error

        state = next_state
        feature = next_feature
    
    rewards_per_episode.append(total_reward)

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Optionally print progress
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")


# Save final Q-table to disk
with open("q_table_training_env_station_feature_pickup_50.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training finished and Q-table saved to q_table_training_env_station_feature_pickup_50.pkl.")
