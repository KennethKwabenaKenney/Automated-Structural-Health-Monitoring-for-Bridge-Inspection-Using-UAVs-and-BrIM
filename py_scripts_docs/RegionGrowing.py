# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:52:13 2023

@author: kenneyke
"""
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2

# Load the image
root = tk.Tk()
IMAGE = filedialog.askopenfilename(title="Select Data")
root.withdraw() 

image = cv2.imread(IMAGE)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the seed point for region growing
seed_point = (100, 100)

# Define the region growing parameters
neighborhood = 4
new_value = 255
max_iter = 100
flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

# Create a mask image for region growing
mask = np.zeros((gray_image.shape[0] + 2, gray_image.shape[1] + 2), np.uint8)

# Apply region growing to the grayscale image using the seed point and parameters
num_iter, mask, rect, center = cv2.floodFill(gray_image, mask, seed_point, new_value, 0, 0, flags)

# Resize the mask image to match the size of the source image
mask = cv2.resize(mask[1:-1, 1:-1], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Apply the mask to the original image to extract the object
extracted_object = cv2.bitwise_and(image, image, mask=mask)

# Display the extracted object
cv2.imshow('Extracted Object', extracted_object)
cv2.waitKey(0)
cv2.destroyAllWindows()