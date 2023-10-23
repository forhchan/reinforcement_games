from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, observation_space_size : int,
                 action_space_size : int):
        super().__init__()
        self.observation_space_size = observation_space_size
        self.hidden_size = observation_space_size
        
        self.l1 = nn.Linear(in_features=self.observation_space_size, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space_size)
        
    def forward(self, data):
        output = F.sigmoid(self.l1(data))
        output = self.l2(output)
        return output
    