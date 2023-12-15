import random
import numpy as np
import cv2
import time


class SnakeEnv:
    def __init__(self) -> None:
        super(SnakeEnv, self).__init__()
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose different codecs (e.g., 'XVID', 'MP4V', 'MJPG')
        self.out = cv2.VideoWriter('snake_game.avi', fourcc, 30.0, (700, 500))
        
        self.reset()
        
    def render(self, n_games, record):
        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down
        # 0-Left, 1-Straight, 2-Right
        
        # Display Screen
        self.img = np.zeros((500,700,3),dtype='uint8')
        self.img[:, 500:, :] = 255
        
        # Displayd Score
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.img, f'Games : {n_games}', (510,250), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(self.img,f' Score : {self.score}',(500,300), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(self.img,f' Best : {record}',(500,350), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        
        
        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+20,self.apple_position[1]+20),(0,0,255),3)
        
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+20,position[1]+20),(0,255,0),3)
        
        # Takes step after fixed time
        t_end = time.time() + 0.0004
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        
        self.out.write(self.img)
            
        cv2.imshow('a',self.img)
        cv2.waitKey(1)
        
    def _place_food(self):
        self.apple_position = [random.randrange(1,25)*20, random.randrange(1,25)*20]
        
        if self.apple_position in self.snake_position:
            self._place_food()
            
    def is_collision(self, snake_head):
        if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
            return 1
        
        if snake_head in self.snake_position[1:]:
            return 1
        
        return 0
    
    def check_repeat_action(self):
        if self.snake_head in self.snake_head_positions:
            if self.repeat_positions != self.snake_head_positions:
                self.repeat_positions = self.snake_head_positions[self.snake_head_positions.index(self.snake_head):]
            else:
                self.repeat_count += 1
                print(f"repeated : {self.repeat_count}")
            self.snake_head_positions = []
            
        if self.repeat_count >= 2:
            return 1
        
        return 0    
    
    def find_danger(self, head, move_direction):
        # clock_wise starts from left = 0: left, 1:up, 2:right, 3:down
        x, y = head[0], head[1]
        head_danger = list((x, y))
        danger = [0, 0, 0]  # danger [left, straight, right]
        for i in range(3):
            # Turning left
            if i == 0:
                idx = (move_direction -1) % 4
            elif i == 1:
                idx = move_direction
            elif i == 2:
                idx = (move_direction +1) % 4
            if idx == 0:
                head_danger[0] -= 20
            elif idx == 1:
                head_danger[1] -= 20
            elif idx == 2:
                head_danger[0] += 20
            elif idx == 3:
                head_danger[1] += 20
            danger[i] = self.is_collision(head_danger)
            
        return danger

    def get_state(self, move_direction):
        repeat = self.check_repeat_action()
        
        move = [0, 0, 0, 0]
        move[move_direction] = 1
        move_left, move_up, move_right, move_down = move
        danger_left, danger_straight, danger_right = self.find_danger(self.snake_head, move_direction)
        
        food = [0, 0, 0, 0]
        if self.snake_head[0] > self.apple_position[0]:
            food[0] = 1
        if self.snake_head[1] > self.apple_position[1]:
            food[1] = 1
        if self.snake_head[0] < self.apple_position[0]:
            food[2] = 1    
        if self.snake_head[1] < self.apple_position[1]:
            food[3] = 1
                
        food_left, food_down, food_right, food_up = food
        
        
        observation = [danger_left, danger_straight, danger_right, move_left, move_up, move_right, move_down, food_up, food_down, food_left, food_right]
        
        return np.array(observation, dtype=int), repeat
        
    def step(self, action):
        # Change the head position based on the button direction
        
        if action == 0:
            # Turning left
            idx = (self.prev_direction -1) % 4
            move_direction = self.clock_wise_directions[idx]
        
        elif action == 1:
            # Moving forward
            move_direction = self.prev_direction
        
        elif action == 2:
            # Turning right
            idx = (self.prev_direction +1) % 4
            move_direction = self.clock_wise_directions[idx]
            
        self.snake_head_positions.append([self.snake_head[0], self.snake_head[1]])
            
        
        # clock_wise = 0: left, 1:up, 2:right, 3:down
        if move_direction == 0:
            self.snake_head[0] -= 20
        elif move_direction == 1:
            self.snake_head[1] -= 20
        elif move_direction == 2:
            self.snake_head[0] += 20
        elif move_direction == 3:
            self.snake_head[1] += 20
        
        
            
        reward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self._place_food()
            self.score += 1
            reward = 10
            self.snake_position.insert(0,list(self.snake_head))

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
        
        done = False
        # On collision kill the snake and print the score
        if self.is_collision(self.snake_head) == 1:
            done = True
            reward = -10
            return reward, done
        
        self.prev_direction = move_direction
        
        return reward, done
    
    def reset(self):
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [[260,260],[240,260],[220,260]]
        self._place_food()
        self.score = 0
        self.clock_wise_directions = [0, 1, 2, 3]  # left, up, right, down
        self.prev_direction = self.clock_wise_directions[2]
        self.snake_head = [260,260]
        self.snake_head_positions = []
        self.repeat_positions = []
        self.repeat_count = 0
        
        # initial_observation = np.zeros(6)  # 적절한 초기값으로 변경해야 합니다.
        
    
