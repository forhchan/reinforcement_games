
import numpy as np
import cv2
import time


def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=2):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

board = np.zeros((16, 4))
char_position = 0
destination_position = 15
trap_positions = [3, 5, 13]
score = 0
prev_button_direction = 1

img = np.zeros((400, 400, 3))


while True:
    cv2.imshow('a',img)
    cv2.waitKey(1)
    img = np.zeros((400,400,3),dtype='uint8')
    # make grid
    img = draw_grid(img, (4, 4))
    # Character
    char_y, char_x = divmod(char_position, 4)
    cv2.rectangle(img, (char_x*100 + 40, char_y*100 + 40),\
            (char_x*100 + 60, char_y*100 + 60),(0,255,0),-1)
    # Moving Position
    # Display trap
    for position in trap_positions:
        trap_y, trap_x = divmod(position, 4)
        cv2.rectangle(img,(trap_x*100, trap_y*100), \
            (trap_x*100+100, trap_y*100+100),(0,0,255),-1)
        
    # Destination Position
    des_y, des_x = divmod(destination_position, 4)
    cv2.rectangle(img,(des_x*100, des_y*100), \
        (des_x*100+100, des_y*100+100),(255,0,0),-1)
    
    # Takes step after fixed time
    t_end = time.time() + 0.05
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(1)
        else:
            continue
            
    # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
    # a-Left, d-Right, w-Up, s-Down
    button_direction = None

    if k == ord('a') and char_x > 0:
        button_direction = 0
    elif k == ord('d') and char_x < 3:
        button_direction = 1
    elif k == ord('w') and char_y > 0:
        button_direction = 3
    elif k == ord('s') and char_y < 3:
        button_direction = 2
    elif k == ord('q'):
        break


    # Change the head position based on the button direction
    if button_direction == 1:
        board[char_y * 4 + char_x][button_direction] = 1
        char_position += 1
        char_x += 1
        
    elif button_direction == 0:
        board[char_y * 4 + char_x][button_direction] = 1
        char_position -= 1
        char_x -= 1
        
    elif button_direction == 2:
        board[char_y * 4 + char_x][button_direction] = 1
        char_position += 4
        char_y += 1
        
    elif button_direction == 3:
        board[char_y * 4 + char_x][button_direction] = 1
        char_position -= 4
        char_y -= 1
    

    # Increase Snake length on eating apple
    if char_position == destination_position:
        score += 1
     
    # On collision kill the snake and print the score
    if char_position in trap_positions or char_position == destination_position:
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((500,500,3),dtype='uint8')
        cv2.putText(img,'Your Score is {}'.format(score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('a',img)
        cv2.waitKey(0)
        cv2.imwrite('D:/downloads/ii.jpg',img)
        print(board)
        break
        
cv2.destroyAllWindows()