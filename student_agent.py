import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, action_size=6, discretize_bins=10):
        self.action_size = action_size
        self.discretize_bins = discretize_bins
        self.q_table = {}
    
    def discretize_state(self, state):
        """
        Convert continuous state values to discrete bins for Q-table lookup.
        Returns a tuple that can be used as dictionary key.
        """
        # Extract grid size from state for normalization
        max_coord = max(max(state[0:10]), 1.0)
        
        # Discretize taxi and station positions
        discrete_state = []
        for i in range(10):
            # Normalize by max coordinate and discretize
            norm_val = state[i] / max_coord
            bin_val = min(self.discretize_bins - 1, int(norm_val * self.discretize_bins))
            discrete_state.append(bin_val)
        
        # Add obstacle and look flags directly (they're already discrete)
        for i in range(10, 16):
            discrete_state.append(state[i])
        
        return tuple(discrete_state)
    
    def act(self, state):
        """Choose the best action based on Q-values."""
        state_key = self.discretize_state(state)
        
        if state_key in self.q_table:
            return np.argmax(self.q_table[state_key])
        else:
            # If state not in Q-table, use a simple heuristic
            return self._heuristic_action(state)
    
    def _heuristic_action(self, state):
        """Simple rule-based heuristic for unseen states."""
        taxi_row, taxi_col = state[0], state[1]
        # Check if passenger is nearby (look flag)
        passenger_nearby = bool(state[14])
        # Check if destination is nearby (look flag)
        destination_nearby = bool(state[15])
        
        # Check obstacle flags
        obstacle_north = bool(state[10])
        obstacle_south = bool(state[11])
        obstacle_east = bool(state[12])
        obstacle_west = bool(state[13])
        
        # If passenger nearby and not picked up, try pickup
        if passenger_nearby:
            return 4  # Pickup
        
        # If destination nearby and passenger picked up, try dropoff
        if destination_nearby:
            return 5  # Dropoff
        
        # Random movement avoiding obstacles
        valid_actions = []
        if not obstacle_north:
            valid_actions.append(1)  # North
        if not obstacle_south:
            valid_actions.append(0)  # South
        if not obstacle_east:
            valid_actions.append(2)  # East
        if not obstacle_west:
            valid_actions.append(3)  # West
        
        if valid_actions:
            return np.random.choice(valid_actions)
        else:
            return np.random.randint(0, 4)  # Random move if all directions blocked
    
    def load(self, filepath):
        """Load Q-table from file."""
        try:
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Loaded Q-table with {len(self.q_table)} states")
            return True
        except:
            print(f"Couldn't load Q-table from {filepath}")
            return False

# Global agent instance
agent = None

def get_action(obs):
    """
    Function used by the environment to get the action from the agent.
    This is the function that will be called during evaluation.
    """
    global agent
    
    # Initialize agent if not already done
    if agent is None:
        agent = QLearningAgent()
        
        # Try loading from different possible locations
        success = False
        potential_paths = [
            "q_table.pkl",
            "models/q_table_final.pkl",
            "./q_table.pkl",
            "./models/q_table_final.pkl"
        ]
        
        for path in potential_paths:
            if agent.load(path):
                success = True
                break
        
        if not success:
            print("Warning: Could not load Q-table from any location. Using heuristic fallback.")
    
    # Choose action
    return agent.act(obs)