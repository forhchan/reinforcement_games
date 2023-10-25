import random
import numpy as np
import cv2
import time
import torch


class Frozen:
    
    COLORS = [(255, 255, 255),
              (0, 255, 255),
              (0, 0, 255),
              (255, 0, 0),
              (0, 0, 0),
              (0, 255, 0)]
    
    BLOCK_SIZE = 40    
    
    def __init__(self) -> None:
        super().__init__()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can choose different codecs (e.g., 'XVID', 'MP4V', 'MJPG')
        self.out = cv2.VideoWriter('output.avi', fourcc, 30.0, (14*self.BLOCK_SIZE, 8*self.BLOCK_SIZE))
        self.reset()
    
    def check_danger(self, char_pos, screen):
        danger_pos = [char_pos-1, char_pos+1, char_pos+8, char_pos-8]
        for idx ,i in enumerate(danger_pos):
            y, x = divmod(i, 8)
            if 0 <= i <= 63 and screen[y][x] == int(0):
                danger_pos[idx] = 1
            else:
                danger_pos[idx] = 0
                
        return danger_pos
    
    def one_hot(self, char_pos, danger):
        state = np.zeros(64)
        danger_pos = np.array(danger)
        state[char_pos] = 1
        state = np.concatenate((state, danger_pos), axis=0)
        state.resize(1, 68)
        state = torch.tensor(state, dtype=torch.float32)
        return state
                    
    def step(self, action):
        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        char_y, char_x = divmod(self.char_position, 8)
        self.char_position_prev = self.char_position
        
        # Change the head position based on the button direction
        if action == 1 and char_x < 7:
            self.char_position += 1
        
        elif action == 0 and char_x > 0:
            self.char_position -= 1
            
        elif action == 2 and char_y < 7:
            self.char_position += 8
              
        elif action == 3 and char_y > 0:
            self.char_position -= 8

        self.reward = 0
        
        if self.char_position in self.trap_positions:
            self.done = True
            self.reward = -10
                 
        elif self.char_position == self.destination_position:
            self.done = True
            self.reward = 10
            
            
        elif (char_x == 0 and action == 0) or (char_x == 7 and action == 1) or \
            (char_y == 0 and action == 3) or (char_y == 7 and action == 2):
            self.done = True
            self.reward = -10
            
            
         
        if self.char_position not in self.char_position_list:
            self.char_position_list.append(self.char_position)
            
        danger = self.check_danger(self.char_position, self.screen)
        
        state = self.one_hot(self.char_position, danger)
        
        return state, self.reward, self.done
    
    def render(self, ep, success_count, success_rate):
        
        for idx, char in enumerate(self.char_position_list[::-1]):
            if idx == 0:
                self.screen[divmod(char, 8)] = 5
            else:
                self.screen[divmod(char, 8)] = 1
        
        screen = [[self.COLORS[x] for x in row] for row in self.screen]
        screen = np.array(screen).reshape((8, 14, 3)).astype(np.uint8)
        screen = screen[:,:,::-1]
        resize_screen = cv2.resize(screen, (14* self.BLOCK_SIZE, 8*self.BLOCK_SIZE), interpolation=cv2.INTER_NEAREST)

        resize_screen[[i * self.BLOCK_SIZE for i in range(8)],:,:] = 0
        resize_screen[:,[i * self.BLOCK_SIZE for i in range(8)], :] = 0
        
        # Display Score and ...
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resize_screen, f'Games :{ep}',(350,60), font, 0.6,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(resize_screen, f'Success :{success_count}',(350,100), font, 0.6,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(resize_screen, f'Last 100 Games Success Rate',(340,140), font, 0.6,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(resize_screen, f'{success_rate*100:.2f} %',(380,180), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        
        self.out.write(resize_screen)
        cv2.imshow('a', resize_screen)
        cv2.waitKey(1)
        
        # Takes step after fixed time
        t_end = time.time() + 0.001
        while time.time() < t_end:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                continue    
    
    def reset(self):
        self.done = False
        self.screen = np.array([[0]*14 for _ in range(8)])
        self.char_position = 0
        self.char_position_prev = 0
        self.destination_position = 60
        trap_positions = [(y*idx) + random.randint(0, 8) for idx, y in enumerate(range(8))]
        self.trap_positions = [x for x in trap_positions if x not in {self.char_position, self.destination_position}]
        # self.trap_positions = [2, 11, 12, 15, 19, 25, 28, 46, 58, 55, 61]
        self.char_position_list = [0]
        # extra board
        self.screen[:, 8:] = 4
        # destination on board
        self.screen[divmod(self.destination_position, 8)] = 2
        # traps on board
        for trap in self.trap_positions:
            self.screen[divmod(trap, 8)] = 3
            
        self.score = 0
        self.reward = 0
        
        danger = self.check_danger(self.char_position, self.screen)
        
        state = self.one_hot(self.char_position, danger)
        
        return state
    

