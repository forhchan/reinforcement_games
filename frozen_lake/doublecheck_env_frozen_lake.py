from frozen_lake import Frozen
import numpy as np

env = Frozen()
episide = 100
# 보드생성
board = np.zeros((16, 4))
total_reward = 0
discount = 0.99
lr = 0.85

for ep in range(episide):
    done = False
    obs, _ = env.reset()
    e = 1. / ((ep/100) +1)
    
    while not done:
        ######################################################################
        # 보드에 보상이 있으면 그쪽으로 아니면 랜덤
        ######################################################################
        # if np.max(board[obs]):
        #     action = np.argmax(board[obs])
        # else:
        #     action = env.action_space.sample()
        
        ######################################################################
        # e greedy  
        ######################################################################
        # if np.random.rand(1) < e:
        #     action = env.action_space.sample()
        # else:
        #     action = np.argmax(board[obs])
        
        ######################################################################
        # Noise 추가 
        ######################################################################
        
        action = np.argmax(board[env.char_position][:] + np.random.randn(1, 4) / (ep + 1))
        
            
        new_obs, reward, done, truncated, info = env.step(action)
        
        ######################################################################
        # Learning Rate 추가 (마음대로 움지일 때)
        ######################################################################
        board[new_obs][action] = (1-lr) * board[new_obs][action] + \
            lr * (reward + np.max(board[obs]) * discount)
        
        board[new_obs][action] = reward + np.max(board[obs]) * discount
        
        total_reward += reward
        # if reward == 1:
        #     print(f"Total : {total_reward}")
        #     print(board)
        
print(f"acc : {total_reward / episide * 100:.2f}%")
print(board)
        