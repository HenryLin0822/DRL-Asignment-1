import numpy as np
import torch
import torch.nn as nn
import os

# Define the DQN model using PyTorch
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "./models/dqn_model_final.pth"
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

def get_action(obs):
    """
    Select an action based on the current observation.
    This function is called by the environment during evaluation.
    """
    global model, device
    
    # Load model if not already loaded
    if model is None:
        load_model()
    
    # Preprocess state
    processed_state = preprocess_state(obs)
    
    # Convert to PyTorch tensor
    state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(device)
    
    # Get action with highest Q-value
    with torch.no_grad():
        q_values = model(state_tensor)
    
    return q_values.cpu().data.numpy().argmax()