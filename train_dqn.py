import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Import the custom taxi environment.
# Make sure that custom_taxi_env.py is in the same directory or adjust the import accordingly.
from custom_taxi_env import CustomTaxiEnv
from simple_custom_taxi_env import SimpleTaxiEnv

# ----------------------------
# Define the Q-network using PyTorch
# ----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        A simple multi-layer perceptron for approximating Q(s,a).
        Args:
            state_dim (int): Dimension of the state vector.
            action_dim (int): Number of possible actions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)   # First fully connected layer
        self.fc2 = nn.Linear(64, 64)           # Second fully connected layer
        self.fc3 = nn.Linear(64, action_dim)   # Output layer: one Q-value per action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------
# Replay Buffer for Experience Replay
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initializes a replay buffer.
        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Saves a transition into the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions.
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# Feature extractor to convert raw state to feature vector
def extract_features(state):
    taxi_row, taxi_col, s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = state

    features = []
    
    # # Create station list
    # stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]

    # for station in stations:
    #     features.append(station[0] - taxi_row)
    #     features.append(station[1] - taxi_col)
    
    features.append(obstacle_north)
    features.append(obstacle_south)
    features.append(obstacle_east)
    features.append(obstacle_west)

    features.append(passenger_look)
    features.append(destination_look)
    
    return np.array(features, dtype=np.float32)

# ----------------------------
# Main Training Function for DQN
# ----------------------------
def train():
    # Hyperparameters
    num_episodes = 10000            # Total episodes for training
    max_steps = 20000                # Maximum steps per episode
    batch_size = 8                # Batch size for experience replay
    gamma = 0.99                   # Discount factor for future rewards
    learning_rate = 1e-3           # Learning rate for optimizer
    buffer_capacity = 10000        # Replay buffer size
    target_update_freq = 10        # Update target network every N episodes
    epsilon_start = 1.0            # Initial exploration probability
    epsilon_final = 0.01           # Final (minimum) exploration probability
    epsilon_decay = 0.9999          # Decay factor for epsilon

    # Create environment instance
    env = CustomTaxiEnv(fuel_limit=10000)
    state, _ = env.reset()
    state_dim = len(extract_features(state))         # In our environment, state is a fixed-length tuple (e.g., length=16)
    action_dim = 6                 # There are 6 discrete actions: South, North, East, West, Pick Up, Drop Off

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize the policy (online) network and the target network
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Set target network to evaluation mode

    # Optimizer for policy network
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Function to decay epsilon over time (for epsilon-greedy exploration)
    def epsilon_by_frame(frame_idx):
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

    frame_idx = 0  # Total number of steps taken

    epsilon = epsilon_start

    rewards_per_episode = []
    # Main training loop over episodes
    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset environment at the start of each episode
        episode_reward = 0
        epsilon = max(epsilon * epsilon_decay, epsilon_final)

        for step in range(max_steps):
            frame_idx += 1
            # epsilon = epsilon_by_frame(frame_idx)
            

            # Epsilon-greedy action selection:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                state_tensor = torch.FloatTensor(extract_features(state)).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)
            # Store the transition in the replay buffer
            replay_buffer.push(extract_features(state), action, reward, extract_features(next_state), done)
            state = next_state
            episode_reward += reward

            # If the replay buffer has enough samples, perform a learning step.
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # Compute Q(s,a) for current states using the policy network.
                q_values = policy_net(states)
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute the next Q-values using the target network.
                with torch.no_grad():
                    next_q_values = target_net(next_states)
                    next_q_value = next_q_values.max(1)[0]

                # Compute the target Q value using the Bellman equation.
                expected_q_value = rewards + gamma * next_q_value * (1 - dones)

                # Calculate the loss using Mean Squared Error (MSE)
                loss = nn.MSELoss()(q_value, expected_q_value)

                # Optimize the policy network.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break
        
        rewards_per_episode.append(episode_reward)

        # Periodically update the target network with the policy network’s weights.
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode:3d} | Reward: {episode_reward:7.2f} | Epsilon: {epsilon:.3f}")
        # Optionally print progress
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_per_episode[-20:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")

    # Save the trained model's state dictionary.
    torch.save(policy_net.state_dict(), "dqn_taxi_model.pkl")
    print("Training complete. Model saved to dqn_taxi_model.pkl")

# ----------------------------
# Run the training loop
# ----------------------------
if __name__ == '__main__':
    train()
