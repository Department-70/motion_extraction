# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 08:51:08 2024

@author: Debra Hogue
TIMEXA - Temporal Image Motion Extractor and Analyzer
"""

import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageOps
from functools import partial
import numpy as np
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from openpyxl import load_workbook

""" Helper function: Make GIF from images in a folder location """        
def make_gif(frame_folder, class_name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save(class_name+"_results.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)

""" Helper function: Adjusts the alpha/opacity of the image even more - preparing it to be a mask for motion enhancement """ 
def reduce_alpha(image):
    # Convert to grayscale
    img = ImageOps.grayscale(image)
    
    # Convert the image to RGBA mode (if it's not already in RGBA mode)
    img = img.convert("RGBA")

    # Get pixel data
    #pixels = img.load()
    
    width, height = img.size
    colored_img = Image.new('RGBA', (width, height))

    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            # Replace grayscale intensity with vivid colors
            if pixel[0] < 85:
                colored_img.putpixel((x, y), (255, 255, 0, 174))  # Yellow for darker shades
            elif pixel[0] < 170:
                colored_img.putpixel((x, y), (0, 255, 0, 174))  # Green for mid-range shades
            else:
                colored_img.putpixel((x, y), (0, 0, 0, 100))  # Black for lighter shades
                
    # Show the modified image
    #show_image(colored_img)
    
    return colored_img

""" Helper function: MAE evaluation of previous frame's motion vs current frame """    
def calculate_mae(prev_mask, curr_mask):
    # Convert images to numpy arrays for easier computation
    prev_array = np.array(prev_mask)
    curr_array = np.array(curr_mask)

    # Compute the absolute difference between the two masks
    absolute_diff = np.abs(prev_array - curr_array)

    # Calculate the mean absolute error (MAE)
    mae = np.mean(absolute_diff)
    
    return mae

""" Helper function: E_m evaluation of previous frame's motion vs current frame """
def calculate_e_measure_pixelwise(prev_frame, curr_frame, threshold=25):
    """
    Calculate the mean E-measure pixelwise between two frames.
    
    Parameters:
    - prev_mask_img: First frame (numpy array) in grayscale
    - curr_mask_img: Second frame (numpy array) in grayscale
    - alpha: Weight parameter for combining precision and recall (default: 0.5)
    - threshold: Threshold for gradient difference (default: 25)
    
    Returns:
    - mean_e_measure: Mean E-measure value
    """
    # Compute the absolute pixel-wise difference between frames
    abs_diff = np.abs(prev_frame.astype(np.float32) - curr_frame.astype(np.float32))

    # Compute precision and recall based on the threshold
    precision = np.mean(abs_diff < threshold)
    recall = np.mean(curr_frame < threshold)

    # Compute E-measure
    alpha = 0.5
    e_measure = 1 - alpha * (1 - precision) - (1 - alpha) * (1 - recall)

    return e_measure

""" Helper function: structural similarity index (SSIM) evaluation of previous frame's motion vs current frame """
def calculate_ssim (prev_frame, curr_frame):
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

"""
    TIMEXA Application Window
"""
class MotionExtractionApp:
    def __init__(self, master):
        self.root = master
        self.root.title("TIMEXA - Temporal Image Motion Extractor and Analyzer App")

        # Variables
        self.gif_folder_path = tk.StringVar()
        self.num_frames_var = tk.IntVar()

        # GUI components
        self.create_widgets()

    def create_widgets(self):
        # Top bar - Select folder and set time frames
        self.frame_top_bar = tk.Frame(self.root)
        self.frame_top_bar.pack(side=tk.TOP, fill=tk.X)

        self.label_folder = tk.Label(self.frame_top_bar, text="Select Folder:")
        self.label_folder.pack(side=tk.LEFT)

        self.entry_folder = tk.Entry(self.frame_top_bar, width=50, textvariable=self.gif_folder_path)
        self.entry_folder.pack(side=tk.LEFT)

        self.button_browse = tk.Button(self.frame_top_bar, text="Browse", command=self.browse_folder)
        self.button_browse.pack(side=tk.LEFT)

        self.label_num_frames = tk.Label(self.frame_top_bar, text="Number of Frames:")
        self.label_num_frames.pack(side=tk.LEFT)

        self.entry_num_frames = tk.Entry(self.frame_top_bar, width=5, textvariable=self.num_frames_var)
        self.entry_num_frames.pack(side=tk.LEFT)
        self.entry_num_frames.insert(0, "10")  # Default value

        self.button_start = tk.Button(self.frame_top_bar, text="Start", command=self.start_motion_extraction)
        self.button_start.pack(side=tk.LEFT)

        # Results section - Display resulting GIF and metrics
        self.frame_results = tk.Frame(self.root)
        self.frame_results.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left section - Display resulting GIF
        self.label_result = tk.Label(self.frame_results, text="Results:")
        self.label_result.pack(side=tk.TOP)

        # Use a Label to display the resulting GIF
        try:
            self.default_image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))  # Blank (transparent) image
            self.image_result = ImageTk.PhotoImage(self.default_image)
        except Exception as e:
            # Handle the exception (print or show an error message)
            print(f"Error loading default image: {e}")
            self.image_result = None

        self.label_image_result = tk.Label(self.frame_results)
        self.label_image_result.pack(side=tk.LEFT)

        # Right section - Display metrics
        self.label_metrics = tk.Label(self.frame_results, text="Metrics:")
        self.label_metrics.pack(side=tk.BOTTOM)

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        self.gif_folder_path.set(folder_path)

    def start_motion_extraction(self):
        folder_path = self.entry_folder.get()
        num_frames = int(self.entry_num_frames.get())

        if os.path.isdir(folder_path):
            # Call your motion extraction logic and update GUI accordingly
            self.extract_motion_and_update_gui(folder_path, num_frames)
        else:
            # Handle invalid folder path
            messagebox.showerror("Error", "Please select a valid folder path.")
            

    def extract_motion_and_update_gui(self, folder_path, num_frames):
        try:
            folder_path = self.gif_folder_path.get()
            num_user_frames = self.num_frames_var.get() # User defined frames (time setting)
    
            if not folder_path or not os.path.isdir(folder_path):
                messagebox.showerror("Error", "Please select a valid GIF folder.")
                return
            
            # Get the root path
            root_path = os.path.dirname(os.path.abspath(__file__))
            print(f"Root Path: {root_path}")
            
            # Get folder name for image collection
            folder_name = os.path.basename(folder_path)
    
            # Get all of the current gif frames
            frames = [Image.open(image) for image in glob.glob(f"{folder_path}/*.JPG")]
            num_frames = len(frames)
            print(f"Number of frames: {num_frames}")
              
            # print("number of frames: " + str(len(frames)))
            # opacity = 0.5  # Adjust opacity level (0.0 to 1.0)
            alpha = 50
            mask_imgs = defaultdict(list)
            # blended_imgs = []
            index = 0
    
            # Iterate through each frame in the folder
            for f in frames:
                # Step 1 - Invert the colors of each frame
                inverted = ImageOps.invert(f)
                # show_image(inverted)
                 
                # Step 2 - Make the frames half transparent - opacity = 50%
                blended = inverted.convert("RGBA")
                blended.putalpha(alpha)
                semifinal = reduce_alpha(blended)
                semifinal.putalpha(alpha)
                # show_image(blended)
                 
                # Step 3 - add blur
                final = semifinal.filter(ImageFilter.BLUR)
                # show_image(final)
    
                mask_imgs[index].append(final)
                index += 1
                print("index: " + str(index))
    
            # Step 4 - Additional enhancement to see the changes in time better and frame evaluation
            maes = []
            ems = []
            ssims = []
             
            # Iterate through the mask images to process motion
            for key, value in mask_imgs.items():
                if key != 0:
                    prev_key = key - 1
                    prev_mask = mask_imgs[prev_key]
                     
                    # Select the first image from the list
                    prev_mask_img = prev_mask[0]
                     
                    # Calculate MAE between previous and current masks
                    curr_mask_img = mask_imgs[key][0]
                    mae = calculate_mae(np.array(prev_mask_img), np.array(curr_mask_img))
                    maes.append(mae)
                    # print(f"Frame {key}: MAE = {mae}")
                     
                    gray_prev_mask_array = np.array(prev_mask_img.convert('L'))
                    gray_curr_mask_array = np.array(curr_mask_img.convert('L'))
                     
                    em = calculate_e_measure_pixelwise(gray_prev_mask_array, gray_curr_mask_array)
                    ems.append(em)
                    # print(f"Frame {key}: Em = {em}")
                     
                    ssim = calculate_ssim(gray_prev_mask_array, gray_curr_mask_array)
                    ssims.append(ssim)
                    # print(f"Frame {key}: SSIM: {ssim}")
                     
                    # # Put the mask onto the original image
                    # finalM = overlay_images(frames[key], prev_mask_img)
                    # # show_image(finalM)
                     
                    # blended_imgs.append(finalM)
                    # print("blended_imgs size: " + str(len(blended_imgs)))
             
            # Save as an animated gif
            # if blended_imgs:
            #     blended_imgs[0].save(
            #         f"./results/{class_name}_motion_results.gif",
            #         format="GIF",
            #         append_images=blended_imgs[1:],
            #         save_all=True,
            #         duration=100,
            #         loop=0
            #     )
             
            # Averages of metrics
            average_MAE = np.mean(maes)
            print(f"Average MAE = {average_MAE}")
            average_Em = np.mean(ems)
            print(f"Average Em = {average_Em}")
            average_SSIM = np.mean(ssims)
            print(f"Average SSIM = {average_SSIM}")
    
            # return average_MAE, average_Em, average_SSIM
    
            # Update labels with results/metrics
            # Display the resulting GIF in the left section
            gif_path = os.path.join(root_path, f"results\{folder_name}_motion_results.gif")
            self.display_gif(gif_path)
            self.label_metrics.config(
                text=f"Metrics: Average MAE = {average_MAE}, Average Em = {average_Em}, Average SSIM = {average_SSIM}"
            )
        except Exception as e:
            # Catch and print any exceptions to help with debugging
            print(f"Error during motion extraction: {e}")

    def display_gif(self, gif_path):
        gif_image = Image.open(gif_path)
        gif_image = ImageTk.PhotoImage(gif_image)
        self.label_image_result.config(image=gif_image)
        self.label_image_result.image = gif_image  # Keep a reference to avoid garbage collection

if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    
    # Instantiate the MotionExtraction class
    app = MotionExtractionApp(root)
    
    # Start the Tkinter main loop
    root.mainloop()
