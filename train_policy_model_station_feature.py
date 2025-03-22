import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from custom_taxi_env import CustomTaxiEnv
from training_taxi_environment import TrainingTaxiEnv

# -------------------------------
# Feature Extraction Function
# -------------------------------
def extract_features(obs):
    """
    Extract features from an observation in the custom taxi environment.

    Parameters:
    - obs: A tuple with 16 elements containing:
         (taxi_row, taxi_col,
          station0_row, station0_col, station1_row, station1_col,
          station2_row, station2_col, station3_row, station3_col,
          obstacle_north, obstacle_south, obstacle_east, obstacle_west,
          passenger_look, destination_look)
    - grid_size: The size (n) of the grid.

    Returns:
    - A NumPy array of extracted features.
    """
    taxi_row, taxi_col = obs[0], obs[1]
    stations = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
    obstacle_flags = list(obs[10:14])
    passenger_flag = obs[14]
    destination_flag = obs[15]
    
    taxi_row_norm = taxi_row
    taxi_col_norm = taxi_col
    features = []
    
    # for (s_row, s_col) in stations:
    #     diff_row = (s_row - taxi_row)
    #     diff_col = (s_col - taxi_col)
    #     manhattan_dist = (abs(s_row - taxi_row) + abs(s_col - taxi_col))
    #     features.extend([diff_row, diff_col, manhattan_dist])

    station_north = int((taxi_row-1, taxi_col) in stations)
    station_south = int((taxi_row+1, taxi_col) in stations)
    station_east = int((taxi_row, taxi_col+1) in stations)
    station_west = int((taxi_row, taxi_col-1) in stations)

    features.append(station_north)
    features.append(station_south)
    features.append(station_east)
    features.append(station_west)

    features.extend(obstacle_flags)
    features.append(passenger_flag)
    features.append(destination_flag)
    
    return np.array(features) # Feature vector dimension = 2 + 4*3 + 4 + 1 + 1 = 20

# -------------------------------
# Policy Network Definition
# -------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=6):
        """
        A simple MLP policy network.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

# -------------------------------
# Utility: Discount Rewards
# -------------------------------
def discount_rewards(rewards, gamma=0.99):
    """
    Compute discounted rewards.
    """
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        discounted[i] = cumulative
    discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)
    return discounted

# -------------------------------
# Training Loop using REINFORCE
# -------------------------------
def train():
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = TrainingTaxiEnv(min_size=7, max_size=9)
    input_dim = 10     # As per our feature extractor.
    hidden_dim = 64
    output_dim = 6     # Number of possible actions.
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    num_episodes = 2000
    gamma = 0.99

    # Track statistics for monitoring training.
    reward_history = []
    loss_history = []
    steps_history = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        # grid_size = env.grid_size
        done = False
        log_probs = []
        rewards = []
        total_reward = 0
        steps = 0
        
        while not done:
            features = extract_features(obs)
            features_tensor = torch.from_numpy(features).float().to(device)
            
            action_probs = policy_net(features_tensor)
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
            
            obs, reward, done, empty_fuel, _ = env.step(action.item())
            rewards.append(reward)
            total_reward += reward
            steps += 1

        if empty_fuel:
            episode -= 1
            continue

        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        
        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        reward_history.append(total_reward)
        loss_history.append(policy_loss.item())
        steps_history.append(steps)
        
        # Print training statistics every 50 episodes.
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            avg_loss = np.mean(loss_history[-50:])
            avg_steps = np.mean(steps_history[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Avg Steps: {avg_steps:.1f}")
    
    # Save the trained model locally.
    torch.save(policy_net.state_dict(), "PN_station_feat_episode_select_1e3.pkl")
    print("Training complete. Model saved as 'PN_station_feat_episode_select_1e3.pkl'.")

if __name__ == "__main__":
    train()
