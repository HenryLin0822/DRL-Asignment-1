import numpy as np
import pickle

class QNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize with default weights (will be overwritten when loading)
        self.weights1 = np.random.randn(state_size, 64) * 0.1
        self.bias1 = np.zeros(64)
        self.weights2 = np.random.randn(64, 64) * 0.1
        self.bias2 = np.zeros(64)
        self.weights3 = np.random.randn(64, action_size) * 0.1
        self.bias3 = np.zeros(action_size)
    
    def predict(self, state):
        # Forward pass
        x = state
        x = np.dot(x, self.weights1) + self.bias1
        x = np.maximum(0, x)  # ReLU
        x = np.dot(x, self.weights2) + self.bias2
        x = np.maximum(0, x)  # ReLU
        q_values = np.dot(x, self.weights3) + self.bias3
        return q_values
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self.weights1 = weights['weights1']
        self.bias1 = weights['bias1']
        self.weights2 = weights['weights2']
        self.bias2 = weights['bias2']
        self.weights3 = weights['weights3']
        self.bias3 = weights['bias3']

# Global model instance
model = None

def preprocess_state(state):
    """Normalize state values to handle different grid sizes"""
    state = np.array(state, dtype=np.float32)
    
    # Find the maximum coordinate value for normalization
    max_coordinate = max(max(state[0:10]), 1.0)
    
    # Normalize all spatial coordinates
    normalized_state = state.copy()
    normalized_state[0:10] /= max_coordinate
    
    return normalized_state

def get_action(obs):
    """
    Function used by the environment to get the action from the agent.
    """
    global model
    
    # Lazy-load the model on first call
    if model is None:
        state_size = 16
        action_size = 6
        model = QNetwork(state_size, action_size)
        
        try:
            model.load('dqn_model.pkl')
            print("Model loaded successfully")
        except:
            try:
                model.load('models/best_dqn_model.pkl')
                print("Best model loaded successfully")
            except:
                print("Warning: No model found. Using untrained model.")
    
    # Preprocess the observation
    state = preprocess_state(obs)
    
    # Get Q-values and select the best action
    q_values = model.predict(state)
    return np.argmax(q_values)