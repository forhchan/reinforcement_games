from snake_env import SnakeEnv
from snakegame_model import Model

import torch
import torch.nn as nn

import random
import numpy as np
import argparse
from collections import deque

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Cleared Lines')
    plt.plot(scores, label="Cleared Lines")
    plt.plot(mean_scores, label="Last 100 games Mean Cleared Lines")
    
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(f'{mean_scores[-1]:.3f}'))
    
    
    if max(scores) > 0:
        plt.scatter(len(scores) -scores[::-1].index(max(scores)) -1, max(scores), c="red", s=10)
        plt.text(len(scores) - scores[::-1].index(max(scores)) -1, max(scores), str(f'best: {max(scores)}'))
    # plt.text(len(losses)-1, losses[-1], str(f'{losses[-1]:.3f}'))
    
    plt.show(block=False)
    plt.legend(loc='upper left')
    # plt.savefig("model_64x128x128.png")
    plt.pause(.1)

    
    
def get_args():
    parse = argparse.ArgumentParser(description="Snake Game")
    parse.add_argument("--MAX_MEMORY", type=int, default=100_000, help="Maximum memory")
    parse.add_argument("--BATCH_SIZE", type=int, default=1000, help="Number of batches")
    parse.add_argument("--LR", type=float, default=1e-3, help="Learning rate")
    parse.add_argument("--gamma", type=float, default=0.99, help="discount rate")
    
    args = parse.parse_args()
    return args

class Trainer():
    def __init__(self, model, gamma):
        self.model = model
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.OPT = torch.optim.Adam(params=self.model.parameters(), lr=args.LR)
        
    def train_step(self, obs, new_obs, reward, action, done):
        obs = torch.from_numpy(obs).type(torch.FloatTensor)
        new_obs = torch.from_numpy(new_obs).type(torch.FloatTensor)
        reward = torch.tensor(reward, dtype=torch.float)
        # action = torch.tensor(action, dtype=torch.float)
        
        
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(obs, dim=0)
            new_obs = torch.unsqueeze(new_obs, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            action = [action]
            done = (done, )
        
        pred = self.model(obs)
        
        target = pred.clone()

        for idx in range(len(pred)):
            new_Q = reward[idx]
            if not done[idx]:
                with torch.inference_mode():            
                    new_Q = reward[idx] + self.gamma * torch.max(self.model(new_obs[idx]).detach())
            target[idx][action[idx]] = new_Q
        
        self.model.train()
        loss = self.loss_fn(pred, target)
        
        self.OPT.zero_grad()
        loss.backward()
        self.OPT.step()
        
    
    
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # randomness
        self.memory = deque(maxlen=args.MAX_MEMORY)
        self.model = Model(input_size=11)
        self.trainer = Trainer(self.model, gamma=args.gamma)
        
    def remember(self, obs, new_obs, reward, action, done):
        self.memory.append((obs, new_obs, reward, action, done))
    
    def train_long_memory(self):
        if len(self.memory) > args.BATCH_SIZE:
            mini_batch = random.sample(self.memory, args.BATCH_SIZE)
        else:
            mini_batch = self.memory
        
        obss, new_obss, rewards, actions, dones = zip(*mini_batch)
        self.trainer.train_step(np.array(obss), np.array(new_obss), rewards, actions, dones)
    
    def train_short_memory(self, obs, new_obs, reward, action, done):
        self.trainer.train_step(obs, new_obs, reward, action, done)
    
    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            self.model.eval()
            with torch.inference_mode():
                prediction = self.model(state0)
            action = torch.argmax(prediction).item()

        return action
    
    def train(self):
        game = SnakeEnv()
        record = 0
        plot_scores = []
        plot_mean_scores = []
        total_scores = 0
        # break infinite loop 
        game_limit = 0
        # game_stuck = 0
        
        # initial obs, reward, done
        # danger_left, danger_straight, danger_right, move_left, move_up, move_right, move_down, food_up, food_down, food_left, food_right
        
        while True:
            game_limit += 1
            # game.render(self.n_games, record)
            # Decide action
            obs = game.get_state(game.prev_direction)
            action = self.get_action(obs)
            self.model.train()
            # perform move and get new state``
            reward, done = game.step(int(action))
            
            new_obs = game.get_state(game.prev_direction)
            # train short memory
            
            if record < 20 or game.score > 20:
                self.train_short_memory(obs, new_obs, reward, action, done)
            
            # remember
            self.remember(obs, new_obs, reward, action, done)
            
            # if game_unchanged_count % 100 == 0:
            #     print(f"count : {game_unchanged_count}")
            
            if done or game_limit > 2000:
                # train long memery
                if game.score > record:
                    record = game.score
                    # print(f"best score : {record} in game :{self.n_games}")
                    # print(f"memory : {len(self.memory)}")
                
                # if self.n_games % 100 == 0:
                #     print(f"n_games : {self.n_games}, score : {game.score} record : {record}")
                
                plot_scores.append(game.score)
                total_scores += game.score
                mean_score = total_scores / (self.n_games + 1)
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                    
                game.reset()
                self.n_games += 1
                self.train_long_memory()
                game_limit = 0
                

if __name__ == '__main__':
    args = get_args()
    agent = Agent()
    agent.train()

