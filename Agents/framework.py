from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import torch
import random

class ReplayBuffer:
    """The storage space for transitions the agent has experienced."""
    def __init__(self, capacity):
        """Initialize the buffer as a double-ended queue, with capacity as the max length."""
        self.buffer = deque([], maxlen=capacity)

    def push(self, transition):
        """Add a transition as a list to the buffer."""
        self.buffer.append(tuple([*transition]))

    def sample(self, batch_size):
        """Take and return a random sample of specified size from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current length of the buffer."""
        return len(self.buffer)

class Agent(ABC, ReplayBuffer):
    """The reinforcement learning agent, which has a replay buffer for experience gathering."""
    def __init__(self, buffer_capacity: int):
        """Initialize the agent's buffer with specified max capacity."""
        super().__init__(buffer_capacity)

    @abstractmethod
    def select_action(self, state):
        """Select an action according to the agent's policy."""
        pass

    @abstractmethod
    def update(self):
        """Update the weights based on the agent's experience in buffer."""
        pass

class FeedForwardNN(torch.nn.Module):
    "The neural network function approximator."
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers=1):
        """Initialize the neural network.
        
        Args:
            input_dim (int): the size of the state space.
            output_dim (int): the size of the action space.
            hidden_dim (int): the number of dimensions in the hidden layer(s).
            hidden_layers (int): the depth of the neural network."""
        super().__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        )
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        """Run x (in this context a state) through the weights of the network and return the predicted output, the reward."""
        x = torch.relu(self.input_layer(x.to(self.device)))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return torch.relu(self.output_layer(x))

    def save(self, filepath):
        """Save the state dictionary to the specified location."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load the state dictionary from the specified file."""
        self.load_state_dict(torch.load(filepath, weights_only=True))

class Featurizer:
    def __init__(self, input_dim):
        self.mean = np.zeros(input_dim)
        self.var = np.ones(input_dim)
        self.count = 0

    def transform_state(self, state):
        state = np.array(state)
        self.count += 1
        self.mean += (state - self.mean) / self.count
        self.var += (state - self.mean) ** 2 / self.count
        normalized_state = (state - self.mean) / np.sqrt(self.var + 1e-8)
        return torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0)
    
    def reset(self):
        self.mean = np.zeros(1)
        self.var = np.ones(1)
        self.count = 0