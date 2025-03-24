import sys
sys.path.append('/usr/local/anaconda3/lib/python3.12/site-packages')

import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random

class TrainingTaxiEnv:
    def __init__(self, min_size=7, max_size=10, obstacle_prob=0.1, fuel_limit=20000):
        """
        A custom Taxi environment that randomizes the grid size (between min_size and max_size),
        passenger start/destination, and obstacle placement each time reset() is called.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.obstacle_prob = obstacle_prob

        self.fuel = fuel_limit
        self.fuel_limit = fuel_limit
        
        # Action meanings (for reference):
        # 0: Move South
        # 1: Move North
        # 2: Move East
        # 3: Move West
        # 4: Pick up passenger
        # 5: Drop off passenger
        
        self.reset()

    def reset(self):
        # Random grid size for this episode
        self.grid_size = random.randint(self.min_size, self.max_size)

        # Randomly place 4 stations on the grid, ensuring they are not adjacent
        self.stations = []
        while len(self.stations) < 4:
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            if (r, c) in self.stations:
                continue
            adjacent = False
            for (sr, sc) in self.stations:
                if abs(sr - r) <= 1 and abs(sc - c) <= 1:
                    adjacent = True
                    break
            if not adjacent:
                self.stations.append((r, c))
        
        # Place random obstacles
        self.obstacles = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.stations:
                    if random.random() < self.obstacle_prob:
                        self.obstacles.add((r,c))
        
        # Randomly choose passenger’s start station and destination station
        self.passenger_loc = random.choice(self.stations)
        possible_dest = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_dest)
        
        # Random initial taxi position that isn’t an obstacle
        while True:
            start_r = random.randint(0, self.grid_size - 1)
            start_c = random.randint(0, self.grid_size - 1)
            if (start_r, start_c) not in self.obstacles and (start_r, start_c) not in self.stations:
                self.taxi_pos = (start_r, start_c)
                break
        
        # Tracking states
        self.passenger_picked = False
        self.fuel = self.fuel_limit  # or whatever limit you want
        
        return self._get_state(), {}

    def step(self, action):
        reward = 0
        empty_fuel = False
        wrong_drop = False
        done = False
        r, c = self.taxi_pos
        nr, nc = r, c

        if action == 0:  # South
            nr += 1
        elif action == 1:  # North
            nr -= 1
        elif action == 2:  # East
            nc += 1
        elif action == 3:  # West
            nc -= 1
        elif action == 4:  # PICKUP
            if self.taxi_pos == self.passenger_loc and not self.passenger_picked:
                # Correct pickup
                self.passenger_picked = True
                reward += 50
            else:
                # Wrong pickup
                reward -= 10
        elif action == 5:  # DROPOFF
            if self.passenger_picked:
                if self.taxi_pos == self.destination:
                    # Successful dropoff\
                    reward += 50
                    done = True
                else:
                    # Wrong dropoff location
                    reward -= 10
                    done = True
                    wrong_drop = True
                self.passenger_picked = False
            else:
                # Dropping off without passenger
                reward -= 10
        else:
            pass  # Invalid action index (shouldn't happen if your code checks)

        # If the action was a movement:
        if action in [0, 1, 2, 3]:
            if (
                nr < 0 or nr >= self.grid_size or
                nc < 0 or nc >= self.grid_size or
                (nr, nc) in self.obstacles
            ):
                # Hit obstacle or went out of bounds
                reward -= 5
            else:
                # Valid move
                self.taxi_pos = (nr, nc)

                if self.taxi_pos in self.stations:
                    reward += 10
                
                # If passenger was in the taxi, passenger location follows taxi
                if self.passenger_picked:
                    self.passenger_loc = self.taxi_pos
        
        # Reduce fuel, check end of episode if out of fuel
        self.fuel -= 1
        if self.fuel <= 0:
            # reward -= 10
            empty_fuel = True
            done = True

        return self._get_state(), reward, done, empty_fuel, wrong_drop, {}

    def _get_state(self):
        """
        Return a tuple that represents the current environment state.
        Minimally includes taxi row/col, passenger row/col (or 'in_taxi'),
        and destination. The simplest approach is a direct numeric encoding.
        """
        (taxi_row, taxi_col) = self.taxi_pos
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        
        # Example: state is just these 6 numbers + a “has_passenger” flag
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        for (sr, sc) in self.stations:
            grid[sr][sc] = 'S'

        # Place obstacles
        for (r, c) in self.obstacles:
            grid[r][c] = 'X'
        
        # Place passenger
        py, px = self.passenger_loc
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
    
        # Place destination
        dy, dx = self.destination
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'

        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = TrainingTaxiEnv()
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = env.stations
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, empty_fuel, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward, step_count

if __name__ == "__main__":
    env_config = {
        # "fuel_limit" : 20000
    }

    agent_scores = []
    step_counts = []
    for _ in range(1):
        agent_score, step_count = run_agent("student_agent.py", env_config, render=True)
        agent_scores.append(agent_score)
        step_counts.append(step_count)
    print(f"Average Score: {np.mean(agent_scores)}")
    print(f"Average Step: {np.mean(step_counts)}")
    print(f"Scores: {agent_scores}")
