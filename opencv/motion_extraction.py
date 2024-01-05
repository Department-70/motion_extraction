import cv2
import numpy as np
import os
import sys

def process_video(input_video_path, output_video_path):

    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    buffer = []
    shifted_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Invert the colors
        inverted_frame = cv2.bitwise_not(frame)

        # get red channel
        red_channel = inverted_frame[:,:,2]
        inverted_red = np.zeros(inverted_frame.shape).astype('uint8')
        inverted_red[:,:,2] = red_channel

        # Create an alpha channel with 50% opacity
        alpha_channel = np.ones((height, width), dtype=frame.dtype) * 127  # 50% of 255
        inverted_frame_with_alpha = cv2.merge((inverted_frame, alpha_channel))
        blurred = cv2.blur(inverted_frame_with_alpha, (width,height))


        ## NEW
        shifted_buffer.append(inverted_frame_with_alpha)
        buffer.append(frame)

    # Define parameters
    frame_pos = 2 # skip one frame to avoid out of bounds errors
    SHIFT_OFFSET = 30
    MAX_FRAMES = len(buffer)
    ALPHA = 0.5
    BETA = 0.5
    while (frame_pos < MAX_FRAMES):

        print("PROCESSING IMAGE ", frame_pos)
        #sys.stdout.write("\033[f")
        out.write(cv2.addWeighted(buffer[frame_pos], ALPHA, shifted_buffer[2][:, :, :3], BETA, 0))
        frame_pos += 1

    # Release everything when done
    cap.release()
    out.release()

    ## NEW
    print(len(buffer))
    print(len(shifted_buffer))

# Example usage
process_video('deer.mp4', 'deer_motion_extract.mp4')
