# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:48:11 2023

@author: kenneyke
"""
import numpy as np
from matplotlib import pyplot as plt
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

# Apply Canny edge detection to obtain edges
canny_edges = cv2.Canny(gray_image, 50, 200)

# Apply a binary threshold to the edges to create a binary mask
threshold_value = 127
max_value = 255
ret, binary_mask = cv2.threshold(canny_edges, threshold_value, max_value, cv2.THRESH_BINARY)

# Apply opening to remove non-straight edges
kernel = np.ones((5,5),np.uint8)
opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# Apply the binary mask to the original image to extract the object
extracted_object = cv2.bitwise_and(image, image, mask=binary_mask)

# Display the extracted object
cv2.imshow('Extracted Object', extracted_object)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(binary_mask)