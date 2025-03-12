import random
import pickle
import numpy as np
from custom_taxi_env import CustomTaxiEnv

# Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999  # Epsilon decays each step or episode

num_episodes = 50000  # Increase as needed
max_steps_per_episode = 200  # Just a safeguard if you want

# Q-Table as a dictionary: {state: [Q-values for each action]}
q_table = {}

def get_q_values(state):
    """Return the Q-values for a given state, initializing if necessary."""
    if state not in q_table:
        # 6 possible actions: [0,1,2,3,4,5]
        q_table[state] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return q_table[state]

env = CustomTaxiEnv(min_size=5, max_size=10, obstacle_prob=0.1)

total_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        # Epsilon-greedy selection
        if random.random() < epsilon:
            action = random.randint(0,5)
        else:
            q_values = get_q_values(state)
            action = int(np.argmax(q_values))
        
        next_state, reward, done, _ = env.step(action)
        
        # Update Q-table
        current_q = get_q_values(state)[action]
        next_q_max = max(get_q_values(next_state))
        new_q = current_q + alpha * (reward + gamma * next_q_max - current_q)
        get_q_values(state)[action] = new_q
        
        state = next_state
        episode_reward += reward
        
        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    total_rewards.append(episode_reward)

    # Optionally print progress
    if (episode+1) % 1000 == 0:
        avg_reward = np.mean(total_rewards[-1000:])
        print(f"Episode {episode+1}, Avg Reward (last 1000): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

# Save final Q-table to disk
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training finished and Q-table saved to q_table.pkl.")
