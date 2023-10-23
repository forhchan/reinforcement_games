import cv2
import numpy as np
import time
import argparse
import random
import torch


# Get all the global variables
def get_args():
    parser = argparse.ArgumentParser("Tetris Reinforece Learnging")
    parser.add_argument("--width", type=int, default=10, help="width of")
    parser.add_argument("--height", type=int, default=18, help="width of")
    # parser.add_argument("--width", type=int, default=10, help="width of")
    # parser.add_argument("--width", type=int, default=10, help="width of")
    # parser.add_argument("--width", type=int, default=10, help="width of")
    # parser.add_argument("--width", type=int, default=10, help="width of")
    
    args = parser.parse_args()
    return args

class Tetris:

    # block colors in order
    block_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255),
        (0, 255, 0)
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

    def __init__(self, width, height):
        
        self.width = width
        self.height = height
        self.block_bags = []
        self.colors = []
        self.block_size = 35
        self.board = [[0] * self.width for _ in range(self.height)]
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([200, 255, 200], dtype=np.uint8)
        self.reset()      
        print(self.height, len(self.board))
        
    def reset(self):
        self.block_nums = list(range(len(self.blocks)))
        random.shuffle(self.block_nums)
        self.block_idx = self.block_nums.pop()
        self.pos = {"x": self.width // 2 - (len(self.blocks[self.block_idx][:]) // 2), "y": 0}
        self.curr_block = self.blocks[self.block_idx]
        # self.frame()
        
    def rotate(self, block):
        rotated_block = [[row[i] for row in block[::-1]] for i in range(len(block[0]))]
        return rotated_block
    
    def store(self, block):
        board = [x[:] for x in self.board]
        for y in range(len(block)):
            for x in range(len(block[0])):
                if block[y][x] and not board[self.pos["y"] + y][self.pos["x"] + x]:
                    board[y+self.pos["y"]][x+self.pos["x"]] = block[y][x]     
        
        return board

    def check_collision(self, block):
        for y in range(len(block)):
            for x in range(len(block[0])):
                if block[y][x] and self.board[self.pos["y"] + 1 + y][self.pos["x"] + x]:
                    return True
        return False
        
    def clear_line(self, board, lines):
        for line in lines[::-1]:
            del board[line]
            board = [[0 for _ in range(self.width)]] + board
        
        return board
    
    def game_over(self):
        pass
    
    def count_holes(self, board):
        num_holes = 0
        # from the top, check if there is any block
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            
            # if there is a block, check the num of holes from it.
            num_holes += len([x for x in col[row+1 :] if x == 0])
        return num_holes
    
    def check_cleared_rows(self, board) -> (int, list):
        to_del = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                # add the line number when the line is full of block
                to_del.append(len(board)-1 - i)
        # del the full lines 
        if len(to_del) > 0:
            board = self.clear_line(board, to_del)
            
        return len(to_del), board
    
    def get_bumpiness_and_height(self, board):
        board = np.array(board)  # to numpy
        mask = board != 0  # mask (if not 0) -> True
        
        # if True get np.argmax(mask. axis=0) else self.height
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights # 15 - [1, 2, 3] - > [14, 13, 12]
        
        total_height = np.sum(heights)
        currs = heights[:-1] # without last one
        nexts = heights[1:] # without first one
        diffs = np.abs(currs -nexts) # each diff before and after ones
        total_bumpiness = np.sum(diffs)
        
        return total_bumpiness, total_height
    
    def get_state(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.count_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])
    
    def get_state_next(self):
        states = {}
        block_idx = self.block_idx
        curr_block = [row[:] for row in self.block]
        if block_idx == 0:
            num_rotations = 1
        elif block_idx == 2 or block_idx == 3 or block_idx == 4:
            num_rotations = 2
        else:
            num_rotations = 4
        
        for i in range(num_rotations):
            valid_xs = self.width - len(curr_block[0])
            for x in range(valid_xs + 1):
                block = [row[:] for row in curr_block]
                self.pos["x"] = x
                while not self.check_collision(block):
                    self.pos["y"] += 1
                board = self.store(block)
                states[(x, i)] = self.get_state(board)
            curr_block = self.rotate(curr_block)
        
        return states
     
    def render(self):
        board = self.store(self.curr_block)
        board = [self.block_colors[p] for row in board for p in row]
        
        # Reshaping data for cv2 - changing the orders [height, widht, demetions]
        img = np.array(board).reshape((self.height, self.width , 3)).astype(np.uint8)
        
        img = img[:, :, ::-1]  # BGR2RGB
        # img = Image.fromarray(img, "RGB")
        # resize_img = img.resize((self.width*self.block_size, self.height*self.block_size), 0)
        # img = np.array(resize_img)
        
        img = cv2.resize(img, (img.shape[1] * self.block_size, img.shape[0] * self.block_size), interpolation=cv2.INTER_NEAREST)
        
        img[[i * self.block_size for i in range(self.height)], :, :] = 0 # make line
        img[:, [i * self.block_size for i in range(self.width)], :] = 0
        img = np.concatenate((img, self.extra_board), axis=1) # add the extra board
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'Your Score is ',(10*self.block_size,8*self.block_size), font, 0.5,(0,0,0),2,cv2.LINE_AA)
        cv2.imshow('a', img)
    
        # cv2.waitKey(1)
        t_end = time.time() + 0.2
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        
        
        if k == ord('a') and self.pos["x"] > 0:
            self.pos["x"] -= 1
        elif k == ord('d') and self.pos["x"] < self.width - len(self.curr_block[0]):
            self.pos["x"] += 1
            
        elif k == ord('s'):
            if not self.check_collision(self.curr_block):
                self.pos["y"] += 1
            else:
                self.board = self.store(self.curr_block)
                self.reset()
                
        elif k == ord('w'):
            self.curr_block = self.rotate(self.curr_block)
            
        elif k == ord('r'):
            while self.pos["y"] < self.height - len(self.curr_block) and not self.check_collision(self.curr_block):
                self.pos["y"] += 1
            self.board = self.store(self.curr_block)
            _, self.board = self.check_cleared_rows(self.board)
            self.reset()
        
    def main(self):
        while True:
            self.render()
            
if __name__ == "__main__":
    opt = get_args()
    test = Tetris(width=opt.width, height=opt.height)
    test.main()