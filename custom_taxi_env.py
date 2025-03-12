import random
import numpy as np

class CustomTaxiEnv:
    def __init__(self, min_size=5, max_size=10, obstacle_prob=0.1):
        """
        A custom Taxi environment that randomizes the grid size (between min_size and max_size),
        passenger start/destination, and obstacle placement each time reset() is called.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.obstacle_prob = obstacle_prob
        
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
        
        # Special stations (R, G, Y, B) randomly placed at corners
        self.stations = [
            (0, 0), 
            (0, self.grid_size - 1),
            (self.grid_size - 1, 0),
            (self.grid_size - 1, self.grid_size - 1)
        ]
        
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
            if (start_r, start_c) not in self.obstacles:
                self.taxi_pos = (start_r, start_c)
                break
        
        # Tracking states
        self.passenger_picked = False
        self.fuel = 5000  # or whatever limit you want
        
        return self._get_state(), {}

    def step(self, action):
        reward = 0
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
            else:
                # Wrong pickup
                reward -= 10
        elif action == 5:  # DROPOFF
            if self.passenger_picked:
                if self.taxi_pos == self.destination:
                    # Successful dropoff
                    reward += 50
                    done = True
                else:
                    # Wrong dropoff location
                    reward -= 10
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
                
                # If passenger was in the taxi, passenger location follows taxi
                if self.passenger_picked:
                    self.passenger_loc = self.taxi_pos
            
            # Small negative reward for each move
            reward -= 0.1
        
        # Reduce fuel, check end of episode if out of fuel
        self.fuel -= 1
        if self.fuel <= 0:
            reward -= 10
            done = True

        return self._get_state(), reward, done, {}

    def _get_state(self):
        """
        Return a tuple that represents the current environment state.
        Minimally includes taxi row/col, passenger row/col (or 'in_taxi'),
        and destination. The simplest approach is a direct numeric encoding.
        """
        (taxi_r, taxi_c) = self.taxi_pos
        if self.passenger_picked:
            # Represent passenger as in-taxi with a special code
            passenger_r, passenger_c = (-1, -1) 
        else:
            passenger_r, passenger_c = self.passenger_loc
        
        dest_r, dest_c = self.destination
        
        # Example: state is just these 6 numbers + a “has_passenger” flag
        return (
            taxi_r, taxi_c,
            passenger_r, passenger_c,
            dest_r, dest_c,
            int(self.passenger_picked)
        )

    def render(self):
        """
        Optional: you can print out the grid for debugging, 
        but you can omit for faster training.
        """
        pass
