# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
from collections import defaultdict

# Global agent instance
_q_table = None

def get_state_key(observation):
    """
    Convert the observation into a hashable state representation.
    """
    taxi_row, taxi_col, *station_coords, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = observation
    
    # Create a simplified state representation
    # Check if passenger is picked up (if passenger is not visible anywhere, it's in the taxi)
    has_passenger = int(passenger_look == 0)
    
    # Include relative position information and obstacles
    state_key = (
        taxi_row, taxi_col,
        has_passenger,
        passenger_look,
        destination_look,
        obstacle_north, obstacle_south, obstacle_east, obstacle_west
    )
    
    return state_key

def get_action(obs):
    """
    Function to be called by the evaluator.
    This function loads the Q-table and chooses the best action
    for the given observation.
    """
    global _q_table
    
    # Initialize Q-table on first call
    if _q_table is None:
        try:
            # Try to load the Q-table from file
            with open("q_table.pkl", "rb") as f:
                _q_table = pickle.load(f)
            # print("Loaded Q-table successfully")
        except (FileNotFoundError, EOFError):
            # If file doesn't exist or is corrupted, initialize empty dict
            _q_table = {}
            # print("No Q-table found, using random policy")
    
    # Get state key
    state_key = get_state_key(obs)
    
    # If state is in Q-table, choose the best action
    if state_key in _q_table:
        return int(np.argmax(_q_table[state_key]))
    
    # If state is not in Q-table, use a fallback strategy
    
    # FALLBACK STRATEGY 1: Random action
    # return random.choice([0, 1, 2, 3, 4, 5])
    
    # FALLBACK STRATEGY 2: Simple heuristic
    taxi_row, taxi_col, *_, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    # If passenger is visible and not picked up, try to pick up
    if passenger_look == 1:
        return 4  # PICKUP
    
    # If passenger is picked up and at destination, drop off
    if passenger_look == 0 and destination_look == 1:
        return 5  # DROPOFF
    
    # Otherwise, move in a direction that's not blocked
    available_moves = []
    
    # Check which directions are available
    if obstacle_north == 0:
        available_moves.append(1)  # NORTH
    if obstacle_south == 0:
        available_moves.append(0)  # SOUTH
    if obstacle_east == 0:
        available_moves.append(2)  # EAST
    if obstacle_west == 0:
        available_moves.append(3)  # WEST
    
    # If no moves are available (shouldn't happen), pick a random action
    if not available_moves:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Choose a random available move
    return random.choice(available_moves)