import argparse
import random
from collections import deque
from tetris_env import Tetris
from tetris_model import Tetris_Model

import torch
import torch.nn as nn

from helpful import plot


def get_args():
    parser = argparse.ArgumentParser("Tetris Reinforece Learnging")
    parser.add_argument("--width", type=int, default=10, help="width of")
    parser.add_argument("--height", type=int, default=18, help="width of")
    parser.add_argument("--extra_width", type=int, default=6, help="board for scores..")
    parser.add_argument("--lr", type=float, default=1e-3, help="model learning rate")
    parser.add_argument("--MAX_LEN", type=int, default=30000, help="number of epochs")
    parser.add_argument("--num_epochs", type=int, default=920, help="train epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="width of")
    parser.add_argument("--batch_size", type=int, default=3000, help="batch_size")
    # parser.add_argument("--rendering", type=bool, default=False, help="render the game")
    # parser.add_argument("--video_record", type=bool, default=False, help="video recording")
    # parser.add_argument("--plot_display", type=bool, default=False, help="Display results with plot")
    
    # parser.add_argument("--width", type=int, default=10, help="width of")
    
    args = parser.parse_args()
    return args


class Trainer():
    def __init__(self, model, gamma):
        self.model = model
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=opt.lr)
        
    def train_step(self, state, reward, new_state, done):
        reward = torch.tensor(reward, dtype=torch.float)
        
        if state.dim() == 1: # change dimension
            state = torch.unsqueeze(state, dim=0)
            new_state = torch.unsqueeze(new_state, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            done = (done, )
        
        pred = self.model(state)[0] # output is two dimension. get [0]
        target = pred.clone()
        for idx in range(len(pred)):
            new_Q = reward[idx] # the case of game is done
            if not done[idx]:
                new_Q = reward[idx] + self.gamma * self.model(new_state)[0].detach()
            target[idx] = new_Q
        
        loss = self.loss_fn(target, pred)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
                

class Agent:
    def __init__(self, width, height, extra_width, gamma, render=False, video_record=False, plot_display=False):
        self.n_games = 0
        self.epsilon = 0    # randomness
        self.memory = deque(maxlen=opt.MAX_LEN)
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.model = Tetris_Model().to(self.device)
        self.trainer = Trainer(self.model, gamma)
        self.env = Tetris(width, height, extra_width, render, video_record)
        self.plot_display = plot_display
        print(f"Device : {self.device}")
        
    def remember(self, state, reward, new_state, done):
        self.memory.append((state, reward, new_state, done))
    
    def train_long_memory(self, batch_size):
        if len(self.memory) > batch_size:
            mini_batch = random.sample(self.memory, batch_size)
        else:
            mini_batch = self.memory
        
        states, rewards, new_states, dones = zip(*mini_batch)
        
        states = torch.stack(states)
        new_states = torch.stack(new_states)
        
        self.trainer.train_step(states, rewards, new_states, dones)
    
    def train_short_memory(self, state, reward, new_state, done):
        self.trainer.train_step(state, reward, new_state, done)
    
    def get_action(self, next_states):
        self.epsilon = 500 - self.n_games
        
        if random.randint(0, 500) < self.epsilon:
            action = random.randint(0, len(next_states) - 1)
        
        else:
            with torch.inference_mode():
                predictions = self.model(next_states)[:, 0] # get number of predictions
            action = torch.argmax(predictions).item()
        return action
    
    def train(self, num_epochs):
        state = self.env.reset().to(self.device) # the first state
        cleared_lines = []
        mean_cleared_lines = []
        total_cleared_lines = 0
        
        while self.n_games < num_epochs:
            next_steps = self.env.get_next_state()
            
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(self.device)
            
            action = self.get_action(next_states)
            
            self.model.train()
            next_state = next_states[action, :].to(self.device)
            action = next_actions[action]
            
            reward, done = self.env.step(action, n_games=self.n_games)
            self.train_short_memory(state, reward, next_state, done)
            
            self.remember(state, reward, next_state, done)
            
            if done:
                self.n_games += 1
                
                ###################### Plot visualizing #############################
                if self.plot_display:
                    cleared_lines.append(self.env.cleared_lines)
                    last_100_cleared_lines = sum(cleared_lines[-100:]) / 100
                    mean_cleared_lines.append(last_100_cleared_lines)
                    plot(cleared_lines, mean_cleared_lines)
                ######################################################################
                
                # if self.n_games % 50 == 0 or self.env.cleared_lines != 0:
                #     print(f"n_games : {self.n_games}, score : {self.env.score}, cleared_lines : {self.env.cleared_lines}")
                self.env.reset()
                self.train_long_memory(opt.batch_size)
                
            else:
                state = next_state
        
if __name__ == "__main__":
    opt = get_args()
    agent = Agent(width=opt.width, height=opt.height, extra_width=opt.extra_width, gamma=opt.gamma, render=False, video_record=False, plot_display=True)
    agent.train(num_epochs=opt.num_epochs)