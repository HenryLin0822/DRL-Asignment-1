import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os

# Define the DQN model using PyTorch
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
passenger_picked_up = False

# For loop detection
state_history = deque(maxlen=20)  # Store last 20 states
action_history = deque(maxlen=20)  # Store last 20 actions
loops_detected = 0
loop_threshold = 3  # Number of repeated patterns before considering it a loop

def preprocess_state(state):
    """
    Normalize state values to handle variable grid sizes.
    This is important for generalizing to different grid dimensions.
    """
    # Convert to numpy array
    processed_state = np.array(state, dtype=np.float32)
    
    # Find maximum coordinate value for normalization
    max_coord = max(max(processed_state[0:10]), 1.0)
    
    # Normalize spatial coordinates
    processed_state[0:10] = processed_state[0:10] / max_coord
    
    return processed_state

def load_model():
    """Load the DQN model from saved weights"""
    global model, device
    
    # Initialize model
    state_size = 16
    action_size = 6
    model = DQN(state_size, action_size).to(device)
    
    # Try loading from different possible paths
    model_paths = [
        "dqn_model.pth",
        "models/dqn_model_final.pth",
        "./dqn_model.pth",
        "./models/dqn_model_final.pth",
        #"./models/best_dqn_model.pth"
    ]
    
    for path in model_paths:
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"Model loaded successfully from {path}")
            model.eval()  # Set to evaluation mode
            return True
        except:
            continue
    
    print("Warning: Could not load model from any location. Using untrained model.")
    return False

def detect_loop(state, action):
    """
    Detect if the agent is stuck in a loop of repetitive states and actions.
    Returns True if a loop is detected, False otherwise.
    """
    global state_history, action_history, loops_detected
    
    # Add current state and action to history
    state_tuple = tuple(state)
    state_history.append(state_tuple)
    action_history.append(action)
    
    # Only check for loops if we have enough history
    if len(state_history) < 10:
        return False
    
    # Check for common loop patterns
    
    # Pattern 1: Oscillating between two states (back and forth)
    if len(state_history) >= 4:
        if (state_history[-1] == state_history[-3] and 
            state_history[-2] == state_history[-4] and
            action_history[-1] == action_history[-3] and
            action_history[-2] == action_history[-4]):
            loops_detected += 1
            return loops_detected >= loop_threshold
    
    # Pattern 2: Repeating the same action in the same state multiple times
    if len(state_history) >= 3:
        if (state_history[-1] == state_history[-2] == state_history[-3] and
            action_history[-1] == action_history[-2] == action_history[-3]):
            loops_detected += 1
            return loops_detected >= loop_threshold
    
    # Pattern 3: Circular movement (returning to the same state after N steps)
    for cycle_length in range(2, min(6, len(state_history) // 2)):
        is_cycle = True
        for i in range(cycle_length):
            if state_history[-(i+1)] != state_history[-(i+1+cycle_length)]:
                is_cycle = False
                break
        if is_cycle:
            loops_detected += 1
            return loops_detected >= loop_threshold
    
    # If no loop detected, decrease the counter
    loops_detected = max(0, loops_detected - 1)
    return False

def get_action(obs):
    """
    Select an action based on the current observation.
    This function is called by the environment during evaluation.
    """
    global model, device, passenger_picked_up, state_history, action_history, loops_detected
    
    # Load model if not already loaded
    if model is None:
        load_model()
    
    # Track passenger status based on observation (state[14] is passenger_look)
    if 'passenger_look' in globals():
        last_passenger_look = passenger_look
    else:
        last_passenger_look = None
    
    passenger_look = bool(obs[14])
    
    # When passenger is no longer visible and was previously visible,
    # they've likely been picked up
    if last_passenger_look and not passenger_look:
        passenger_picked_up = True
    
    # Quick heuristic for when no model is available
    if model is None:
        action = heuristic_action(obs)
        
        # Check for loops
        if detect_loop(obs, action):
            print("Loop detected, taking random action")
            action = np.random.randint(0, 6)  # Take a completely random action
            loops_detected = 0  # Reset loop counter
            state_history.clear()  # Clear history
            action_history.clear()
        
        return action
    
    # Preprocess state
    processed_state = preprocess_state(obs)
    
    # Convert to PyTorch tensor
    state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(device)
    
    # Get action with highest Q-value
    with torch.no_grad():
        q_values = model(state_tensor)
    
    # Get best action according to the model
    q_values_np = q_values.cpu().data.numpy()[0]
    
    # If a loop is detected, choose the second-best action instead
    if detect_loop(obs, np.argmax(q_values_np)):
        print("Loop detected, taking alternative action")
        
        # Zero out the highest Q-value
        best_action = np.argmax(q_values_np)
        q_values_np[best_action] = -np.inf
        
        # Choose the second-best action
        action = np.argmax(q_values_np)
        
        # If still in a loop, just take a random action
        if action == best_action or loops_detected > 2*loop_threshold:
            action = np.random.randint(0, 6)
            loops_detected = 0  # Reset loop counter
            state_history.clear()  # Clear history
            action_history.clear()
        
        return action
    
    return np.argmax(q_values_np)

def heuristic_action(obs):
    """Simple rule-based policy as fallback"""
    global passenger_picked_up
    
    # Extract observations
    taxi_row, taxi_col = obs[0], obs[1]
    
    # Check passenger and destination visibility
    passenger_nearby = bool(obs[14])
    destination_nearby = bool(obs[15])
    
    # Check obstacle flags
    obstacle_north = bool(obs[10])
    obstacle_south = bool(obs[11])
    obstacle_east = bool(obs[12])
    obstacle_west = bool(obs[13])
    
    # If passenger nearby and not picked up, try pickup
    if passenger_nearby and not passenger_picked_up:
        return 4  # Pickup
    
    # If passenger picked up and destination nearby, try dropoff
    if passenger_picked_up and destination_nearby:
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