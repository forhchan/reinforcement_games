import cv2
import numpy as np
import time
import random
import torch


class Tetris:

    # block colors in order
    block_colors = [
        (0, 0, 0),  # Background color
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255),
        (0, 255, 0),
        (200, 255, 200) # 9 extra board color
    ]

    # blocks number = block colors indexes
    blocks = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(self, width, height, extra_width):
        
        self.width = width
        self.height = height
        self.extra_width = extra_width
        self.block_bags = []
        self.colors = []
        self.block_size = 35
        self.best_cleared_lines = 0
        self.reset()      
        
        ###################### Video Recording ############################### 
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can choose different codecs (e.g., 'XVID', 'MP4V', 'MJPG')
        # self.out = cv2.VideoWriter('tetris_game.mp4', fourcc, 30.0, ((self.width + 6) * 35, self.height * 35))
        ######################################################################
        
    def reset(self) -> torch.FloatTensor:
        # random choice of blocks
        self.block_nums = list(range(len(self.blocks)))
        random.shuffle(self.block_nums)
        self.block_idx = self.block_nums.pop()
        self.board = [[0] * self.width for _ in range(self.height)]
        
        # set the block on the first screen
        self.pos = {"x": self.width // 2 - (len(self.blocks[self.block_idx][:]) // 2), "y": 0}
        
        self.block = [row[:] for row in self.blocks[self.block_idx]]  
        
        self.gameover = False  # eq done
        self.score = 0
        self.cleared_lines = 0
        
        return self.get_state(self.board)
        
    def rotate(self, block):
        # Turn the block clockwise
        rotated_block = [[row[i] for row in block[::-1]] for i in range(len(block[0]))]
        return rotated_block
    
    def store(self, block, pos):  # setting the board
        board = [x[:] for x in self.board]
        for y in range(len(block)):
            for x in range(len(block[0])):
                if block[y][x] and not board[pos["y"] + y][pos["x"] + x]:
                    board[y+pos["y"]][x+pos["x"]] = block[y][x]     
        
        return board

    def check_collision(self, block, pos):
        for y in range(len(block)):
            for x in range(len(block[0])):
                # check if the block reaches to the bottom
                # check if the block crash the stacked blocks (y+1 position)
                if (pos["y"] + 1 + y) > self.height - 1 or block[y][x] and self.board[pos["y"] + 1 + y][pos["x"] + x]:
                    return True
        return False
        
    def clear_line(self, board, lines) -> list[int]:
        # delete the lines (cleared lines) from the top
        for line in lines:
            del board[line]
            board = [[0 for _ in range(self.width)]] + board
        return board
    
    def game_over(self, pos_x, cur_block):
        block = [row[:] for row in cur_block]
        for y in range(len(block)):
            for x in range(len(block[0])):
                # check if the block where on the top crashes the stacked blocks in every positions of x
                if block[y][x] and self.board[y][pos_x + x]:
                    return True
                
        return False
    
    def count_holes(self, board) -> int:
        num_holes = 0
        # from the top, check if there is any block vertically
        for col in zip(*board): 
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            
            # if there is a block, check the num of holes from it.
            num_holes += len([x for x in col[row+1:] if x == 0])
        return num_holes
    
    def check_cleared_rows(self, board) -> (int, list):
        to_del = []
        # check if there is a line that full with block from the bottom
        for i, row in enumerate(board):
            if 0 not in row:
                to_del.append(i)
                
        # del the full lines 
        if len(to_del) > 0:
            board = self.clear_line(board, to_del)
        return len(to_del), board
    
    def get_bumpiness_and_height(self, board) -> (int, int):
        board = np.array(board)  # to numpy
        mask = board != 0  # mask (if not 0) -> True
        
        # if True get np.argmax(mask. axis=0) else self.height
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights # 15 - [1, 2, 3] - > [14, 13, 12]
        
        total_height = np.sum(heights)
        currs = heights[:-1] # without last one
        nexts = heights[1:] # without first one
        diffs = np.abs(currs -nexts) # each diffes with next one
        total_bumpiness = np.sum(diffs)
        
        return total_bumpiness, total_height
    
    def get_state(self, board) -> torch.FloatTensor:
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.count_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])
    
    def get_next_state(self) -> dict[(int, int): torch.FloatTensor]:
        states = {}
        block_idx = self.block_idx
        curr_block = [row[:] for row in self.blocks[block_idx]]
        
        if block_idx == 0:
            num_rotations = 1
        elif block_idx == 2 or block_idx == 3 or block_idx == 4:
            num_rotations = 2
        else:
            num_rotations = 4
            
        for i in range(num_rotations):
            valid_xs = self.width - len(curr_block[0])
            # check all the positions of x from the left to the right
            # get states that each rotation and each x position stacked env
            for x in range(valid_xs + 1):
                block = [row[:] for row in curr_block]
                pos = {"x": x, "y": 0}
                
                # check if the game is over
                if self.game_over(pos["x"], curr_block):
                    board = self.board
                else:
                    while not self.check_collision(block, pos):
                        pos["y"] += 1
                    board = self.store(block, pos)
                    
                states[(x, i)] = self.get_state(board)
            curr_block = self.rotate(curr_block)
        return states
     
    def render(self, n_games=None):
        board = [row[:] for row in self.board]
        if self.pos["y"] == 0:
            board = self.update_curr_board(board)
        
        extra_board = np.array([[9] * self.extra_width for _ in range(self.height)])
        # add preview blocks in extra_board
        for y in range(len(self.block)):
            for x in range(len(self.block[0])):
                if self.block[y][x] == 0:
                    extra_board[y+2][x+1] = 9
                else:
                    extra_board[y+2][x+1] = self.block[y][x]
        
        board = np.concatenate((np.array(board), extra_board), axis=1)
        board = [self.block_colors[p] for row in board for p in row]
        img = np.array(board).reshape((self.height, (self.width + self.extra_width), 3)).astype(np.uint8)
        img = img[:, :, ::-1]
        img = cv2.resize(img, ((self.width + self.extra_width)* self.block_size, self.height*self.block_size), interpolation=cv2.INTER_NEAREST)
        
        ####################### Another way ###################################
        # 
        # board = [self.block_colors[p] for row in board for p in row]
        # # Reshaping data for cv2 - changing the orders [height, widht, demetions]
        # img = np.array(board).reshape((self.height, self.width , 3)).astype(np.uint8)
        # img = img[:, :, ::-1]  # BGR2RGB
        # img = Image.fromarray(img, "RGB")
        # resize_img = img.resize((self.width*self.block_size, self.height*self.block_size), 0)
        # img = np.array(resize_img)
        ######################################################################
        
        img[[i * self.block_size for i in range(self.height)], :self.width*self.block_size, :] = 0 # make line
        img[[(i * self.block_size) +1 for i in range(self.height)], :self.width*self.block_size, :] = 0 # make line
        img[:, [i * self.block_size for i in range(self.width)], :] = 0
        img[:, [(i * self.block_size) +1 for i in range(self.width)], :] = 0
        # img = np.concatenate((img, self.extra_board), axis=1) # add the extra board
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.cleared_lines > self.best_cleared_lines:
            self.best_cleared_lines = self.cleared_lines
        cv2.putText(img, f'Next Block',(11*self.block_size,1*self.block_size), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img, f'Games : {n_games} ',(10*self.block_size,8*self.block_size), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img, f'Score : {self.score}',(10*self.block_size,9*self.block_size), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img, f'Best Cleared ',(10*self.block_size,10*self.block_size), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img, f'{self.best_cleared_lines}',(11*self.block_size,11*self.block_size), font, 0.8,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(img, f'cleared_lines ',(10*self.block_size,12*self.block_size), font, 0.8,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(img, f'{self.cleared_lines}',(11*self.block_size,13*self.block_size), font, 0.8,(0,255,0),2,cv2.LINE_AA)
        
        
        ###################### Video Recording  #############################
        # self.out.write(img)
        ######################################################################
        
        cv2.imshow('a', img)
    
        t_end = time.time() + 0.00002
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            elif k == ord('q'):
                self.out.release()
                cv2.destroyAllWindows()
                        
        # if k == ord('a') and self.pos["x"] > 0:
        #     self.pos["x"] -= 1
        # elif k == ord('d') and self.pos["x"] < self.width - len(self.block[0]):
        #     self.pos["x"] += 1
            
        # elif k == ord('s'):
        #     if not self.check_collision(self.block):
        #         self.pos["y"] += 1
        #     else:
        #         self.board = self.store(self.block)
        #         self.reset()
                
        # elif k == ord('w'):
        #     self.block = self.rotate(self.block)
            
        # elif k == ord('r'):
        #     while self.pos["y"] < self.height - len(self.block) and not self.check_collision(self.block):
        #         self.pos["y"] += 1
        #     self.board = self.store(self.block)
        #     _, self.board = self.check_cleared_rows(self.board)
        #     self.game_over()
        #     self.reset()

    def update_curr_board(self, board):
        if self.pos["y"] == 0:
            x_pos = self.width // 2 - (len(self.blocks[self.block_idx][:]) // 2)
        else:
            x_pos = self.pos["x"]
            
        for y in range(len(self.block)):
            for x in range(len(self.block[0])):
                board[y + self.pos["y"]][x + x_pos] = self.block[y][x]
        return board
                
    def new_block(self):
        if not len(self.block_bags):
            self.block_bags = list(range(len(self.blocks)))
            random.shuffle(self.block_bags)
        self.block_idx = self.block_bags.pop()
        self.block = [row[:] for row in self.blocks[self.block_idx]]
        self.pos = {"x": self.width // 2 - (len(self.blocks[self.block_idx][:]) // 2), "y": 0}
        
        if self.check_collision(self.block, self.pos):
            self.gameover = True
            
    def step(self, action, render=False, n_games=None):
        x, num_rotations = action
        self.pos = {"x": x, "y": 0}
        # rotate the block in number of num_rotations
        for _ in range(num_rotations):
            self.block = self.rotate(self.block)
        
        if self.game_over(self.pos["x"], self.block):
            self.gameover = True
            
        else:
            while not self.check_collision(self.block, self.pos):
                if render:
                    self.render(n_games=n_games)
                self.pos["y"] += 1
                
            self.board = self.store(self.block, self.pos)        
        
        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.cleared_lines += lines_cleared
        
        if not self.gameover:
            self.new_block()
        
        if self.gameover:
            self.board = [[0] * self.width for _ in range(self.height)]
            self.new_block()
            score -= 10

        return score, self.gameover
        