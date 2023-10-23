from typing import Any
from snake_env import SnakeEnv
import numpy as np
import random
from collections import deque

env = SnakeEnv()

for _ in range(100):
    action = env.action_space.sample()
    new_obs, reward, done, _, _ = env.step(action)
    
    
# MAX_MEMORY = 100_000
# env = SnakeEnv()
# episide = 100
# # 보드생성
# board = np.zeros((16, 4))
# total_reward = 0
# discount = 0.99
# lr = 0.85

# class Agent:
    
#     def __init__(self):
#         self.n_games = 0
#         self.epsilon = 0 # randomness
#         self.gamma = 0 #discount rate
#         self.memory = deque(maxlen=MAX_MEMORY)
    
#     def get_state(self, game):
#         pass
    
#     def remember(self, state, action, reward, next_state, done):
#         pass
    
#     def train_long_memory(self):
#         pass
    
#     def get_action(self):
#         pass
    

# def train():
#     plot_scores = []
#     plot_mean_score = []
#     total_score = 0
#     record = 0
#     agent = Agent()
#     game = SnakeEnv()
#     done = False
    
#     while not done:
#         # get old state
#         state_old = agent.get_state(game)
        
#         # get move
#         final_move = agent.get_action(state_old)
        
#         # perform move and get new state
#         obs, reward, done, _, _ = game.step(final_move)
#         state_new = agent.get_state(game)
        
#         # train short memory
#         agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
#         # remember
#         agent.remember(state_old, final_move, reward, state_new, done)
        
#         if done:
#             # train long memory, plot result
#             game.reset()
#             agent.n_games += 1
#             agent.train_long_memory()
            
#             if score > record:
#                 record = socre
            
#             print(f'Game : {agent.n_games}, Score : {score}, Record :' {record})        
    
        
# print(f"acc : {total_reward / episide * 100:.2f}%")
# print(board)
        