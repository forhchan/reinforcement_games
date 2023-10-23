from frozen_model import Model
from frozen_lake_env import Frozen
import torch
import numpy as np
import torch.nn.functional as F
import random
from collections import deque
# import wandb

# run = wandb.init(project="frozen_test")
conf = {
    "learning_rate" : 1e-4,
    "epochs": 3000,
    "batch_size": 100
}

class Agent:
    def __init__(self):
        self.model = Model(observation_space_size=68,
                           action_space_size=4)
        
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        # self.agent = self.agent.to(self.device)
            
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        self.success = []
        self.jList = []
        self.frozen = Frozen()
        self.total_loss = []
        self.success_count = 0
        self.success_rate = deque(maxlen=100)
        
    def get_action(self, state, success_rate):
        if np.random.rand(1) < (0.1 - (success_rate / 10)):
            # action = np.random.choice(direction)
            action = random.randint(0, 3)
        else:
            output = self.model(state) 
            # for i in no_directions:
            #     output[0][i] = -10
            _, idx = torch.max(output, 1)
            a = idx.numpy()   
            action = a[0]
            
        return action
            
    
    def train(self, epoch):
        if epoch <= 0:
            raise ValueError('epoch must be positive integer')
            
        # randon epsilon
        rand = 0.01
        rand_acc = -(rand / epoch)
        s_rate = 0
        
        for ep in range(epoch):
            state = self.frozen.reset()
            j = 0
            losses = 0    
            while j < 1000:
                
                action = self.get_action(state, s_rate)
                
                self.frozen.render(ep, self.success_count, s_rate)
                new_state, reward, done = self.frozen.step(action)
                
                # calculate target and loss
                
                target_q = reward + 0.99 * torch.max(self.model(new_state).detach()) # detach from the computing flow
                loss = F.smooth_l1_loss(self.model(state)[0][action], target_q)
                losses += loss.item()
                # update model to optimize Q
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # update state
                state = new_state
                j += 1
                if done:
                    break
                    
            # update EPS
            rand += rand_acc
            
            # append results onto report lists
            if done and reward > 0:
                self.success.append(1)
                self.success_count += 1
                self.success_rate.append(1)
            else:
                self.success.append(0)
                self.success_rate.append(0)
            
            s_rate = sum(self.success_rate) / 100
            self.jList.append(j)
            # wandb.log({"loss" : losses // j}, step=ep)
        print("last 100 epoches success rate: " + str(sum(self.success[-100:])) + "%")
        return self.total_loss
    
if __name__ == "__main__":
    a = Agent()
    a.train(2000)
