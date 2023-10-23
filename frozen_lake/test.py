import cv2
import numpy as np

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose different codecs (e.g., 'XVID', 'MP4V', 'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))  # 'output.avi' is the output video file name

# Generate a sequence of frames (for demonstration, you can replace this with your own frames)
for i in range(100):
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # Generate a random frame as an example
    out.write(frame)  # Write the frame to the video file

# Release the VideoWriter and close the output file
out.release()