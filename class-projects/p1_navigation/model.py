import torch 
import torch.nn.functional as F
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(0)
        self.hidden_sizes = [128, 64] 
        self.input_size = state_size
        self.output_size = action_size
        self.fc1 = nn.Linear(state_size, self.hidden_sizes[0])
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.output = nn.Linear(self.hidden_sizes[1], action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = F.relu(self.output(x))
        return out
        
        