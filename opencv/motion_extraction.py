import cv2
import numpy as np
import os
import sys

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(prev_frame, curr_frame):
    """
    Calculate the structural similarity index (SSIM) between two sequential frames.
    Parameters:
    - prev_mask_img: First frame (numpy array) in grayscale
    - curr_mask_img: Second frame (numpy array)) in grayscale
    Returns:
    - mean_e_measure: Mean E-measure value
    """
    # Calculate SSIM between two frames
    ssim_index, _ = ssim(prev_frame, curr_frame, full=True)
    return ssim_index

def calculate_mae(prev_mask, curr_mask):
    # Convert images to numpy arrays for easier computation
    prev_array = np.array(prev_mask)
    curr_array = np.array(curr_mask)
    # Compute the absolute difference between the two masks
    absolute_diff = np.abs(prev_array - curr_array)
    # Calculate the mean absolute error (MAE)
    mae = np.mean(absolute_diff)
    return mae

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
    frame_number = 0

    while cap.isOpened():
        print("Processing frame: ", frame_number)
        frame_number += 1
        try:
            ret, frame = cap.read()
        except:
            print("An excption occurred...")       

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

        ## Append the frames to the buffers
        shifted_buffer.append(inverted_frame_with_alpha)
        buffer.append(frame)

    # Define parameters
    frame_pos = 0 # skip one frame to avoid out of bounds errors
    SHIFT_OFFSET = 15
    MAX_FRAMES = len(buffer)
    ALPHA = 0.5
    BETA = 0.5

    ssim = 0.
    mae = 0.

    # Print the details because something is weird.
    print('NUMBER OF FRAMES: ', MAX_FRAMES)
    print('SHIFT OFFSET: ', SHIFT_OFFSET)
    print('ALPHA: ', ALPHA)
    print('BETA: ', BETA)

    while (frame_pos < MAX_FRAMES):

        print("PROCESSING IMAGE ", frame_pos)
        #sys.stdout.write("\033[f")

        std = buffer[frame_pos]
        mask = shifted_buffer[frame_pos - SHIFT_OFFSET][:, :, :3]
        new_ssim = calculate_ssim(mask, std)
        new_mae = calculate_mae(mask, std)
        ssim += new_ssim
        mae += new_mae

        out.write(cv2.addWeighted(std, ALPHA, mask, BETA, 0))
        frame_pos += 1

    # Release everything when done
    cap.release()
    out.release()

    ## NEW
    print('SSIM Value: ', ssim/frame_pos)
    print('MAE Value: ', mae/frame_pos)

# Example usage
FILE_NAME = 'rain4.m4v'
process_video(FILE_NAME, 'extract_' + FILE_NAME)