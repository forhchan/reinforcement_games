import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=3):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )
        
        
        # self.fn = nn.Sequential(
        #     nn.Linear(in_features=input_size, out_features=128),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(in_features=hidden_size, out_features=output_size)
        # )
        
        
        
    def forward(self, x):
        x = self.fn(x)
        return x


                


        
            