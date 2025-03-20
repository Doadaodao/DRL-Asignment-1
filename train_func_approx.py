import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
import os
import sys

# Add necessary path for imports
sys.path.append('/usr/local/anaconda3/lib/python3.12/site-packages')

# Import the custom environment
from custom_taxi_env import CustomTaxiEnv

# Define neural network for Q-function approximation
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Feature extractor to convert raw state to feature vector
def extract_features(state):
    taxi_row, taxi_col, s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = state
    
    # Create station list
    stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]
    
    # Find passenger location (assuming it's in one of the stations)
    # This is an approximation since we don't directly know which station has the passenger
    passenger_station = None
    destination_station = None
    
    for station in stations:
        # Check if station is adjacent to the taxi or at the taxi position
        is_adjacent = (abs(station[0] - taxi_row) <= 1 and abs(station[1] - taxi_col) <= 1)
        if is_adjacent and passenger_look == 1:
            passenger_station = station
        if is_adjacent and destination_look == 1:
            destination_station = station
    
    # If we couldn't determine stations, use fallbacks
    if passenger_station is None:
        # If passenger_look is 1, passenger might be at taxi position
        if passenger_look == 1:
            passenger_station = (taxi_row, taxi_col)
        else:
            # Default to the one randomly chosed station
            passenger_station = random.choice(stations)


    if destination_station is None:
        # If destination_look is 1, destination might be at taxi position
        if destination_look == 1:
            destination_station = (taxi_row, taxi_col)
        else:
            # Default to a station that's not the passenger station
            for station in stations:
                if station != passenger_station:
                    destination_station = station
                    break
    
    # Calculate Manhattan distances
    dist_to_passenger = abs(taxi_row - passenger_station[0]) + abs(taxi_col - passenger_station[1])
    dist_passenger_to_dest = abs(passenger_station[0] - destination_station[0]) + abs(passenger_station[1] - destination_station[1])
    
    # Calculate normalized relative positions
    rel_passenger_row = (passenger_station[0] - taxi_row) / 10.0  # Normalized by max grid size
    rel_passenger_col = (passenger_station[1] - taxi_col) / 10.0
    rel_dest_row = (destination_station[0] - taxi_row) / 10.0
    rel_dest_col = (destination_station[1] - taxi_col) / 10.0
    
    # Passenger status (simplified: 1 if passenger is at taxi position, else 0)
    passenger_at_taxi = float(passenger_look == 1 and (taxi_row, taxi_col) == passenger_station)
    
    # Destination status
    at_destination = float(destination_look == 1 and (taxi_row, taxi_col) == destination_station)
    
    # Obstacle information (already binary)
    obstacles = [obstacle_north, obstacle_south, obstacle_east, obstacle_west]
    
    # Taxi position normalized by max grid size
    norm_taxi_row = taxi_row / 10.0
    norm_taxi_col = taxi_col / 10.0
    
    # Combine features
    features = [
        norm_taxi_row,
        norm_taxi_col,
        rel_passenger_row,
        rel_passenger_col,
        rel_dest_row,
        rel_dest_col,
        dist_to_passenger / 20.0,  # Normalize by max possible distance
        dist_passenger_to_dest / 20.0,
        passenger_at_taxi,
        at_destination,
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    ]
    
    return np.array(features, dtype=np.float32)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=64, target_update=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Create Q networks (policy and target)
        self.policy_net = QNetwork(state_size, hidden_size, action_size)
        self.target_net = QNetwork(state_size, hidden_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Create loss function
        self.criterion = nn.MSELoss()
    
    def get_action(self, state, training=True):
        state_feature = extract_features(state)
        state_tensor = torch.FloatTensor(state_feature).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample mini-batch from memory
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Extract batch components
        states = np.array([extract_features(s) for s in batch[0]])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array([extract_features(s) for s in batch[3]])
        dones = np.array(batch[4])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Calculate current Q values
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        
        # Calculate target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q
        
        # Calculate loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, filename):
        torch.save({
            'policy_model': self.policy_net.state_dict(),
            'target_model': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filename)
        
        # Also save a pickle file with the extracted model for easy loading in student_agent.py
        model_data = {
            'policy_model': self.policy_net.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }
        with open(filename.replace('.pth', '.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_model'])
        self.target_net.load_state_dict(checkpoint['target_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

# Training function
def train_agent(episodes=3000, hidden_size=128, learning_rate=0.001, render=False):
    env = CustomTaxiEnv()
    state_size = 16  # Number of features from extract_features function
    action_size = 6  # 6 possible actions
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate
    )
    
    # Track scores
    scores = []
    recent_scores = []
    
    print("Starting training...")
    
    for episode in range(1, episodes+1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.replay()
            
            # Move to next state
            state = next_state
            total_reward += reward
            
            # Render if specified
            if render:
                taxi_row, taxi_col, *_ = state
                env.render_env((taxi_row, taxi_col))
        
        # Update epsilon
        agent.update_epsilon()
        
        # Track scores
        scores.append(total_reward)
        recent_scores.append(total_reward)
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(recent_scores)
            print(f"Episode: {episode}/{episodes}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save checkpoint every 1000 episodes
        if episode % 1000 == 0:
            agent.save(f"taxi_dqn_checkpoint_{episode}.pth")
    
    # Save final model
    agent.save("taxi_dqn_final.pth")
    
    print("Training complete!")
    return agent, scores

# Run training with hyperparameters
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Train the agent
    agent, scores = train_agent(episodes=3000, hidden_size=256, learning_rate=0.001)
    
    # Plot training performance
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('DQN Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig('training_performance.png')
    plt.show()