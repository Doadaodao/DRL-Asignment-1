import random
import pickle
import numpy as np

from custom_taxi_env import CustomTaxiEnv
from simple_custom_taxi_env import SimpleTaxiEnv

# Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_min = 0.1
epsilon_decay = 0.9994  # Epsilon decays each step or episode

num_episodes = 5000  # Increase as needed
max_steps_per_episode = 4000  # Just a safeguard if you want

# Q-Table as a dictionary: {state: [Q-values for each action]}
q_table = {}

def get_q_values(state):
    """Return the Q-values for a given state, initializing if necessary."""
    if state not in q_table:
        # 6 possible actions: [0,1,2,3,4,5]
        q_table[state] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return q_table[state]

env = CustomTaxiEnv()

rewards_per_episode = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    done = False

    while not done:
        # Initialize the state in the Q-table if it is not already present.
        if state not in q_table:
            q_table[state] = np.zeros(6)

        # Implement an ε-greedy policy for action selection.
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5])
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if next_state not in q_table:
            q_table[next_state] = np.zeros(6)
        
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error

        state = next_state
    
    rewards_per_episode.append(total_reward)

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Optionally print progress
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")


# Save final Q-table to disk
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training finished and Q-table saved to q_table.pkl.")
