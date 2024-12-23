# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:39:45 2023

@author: kenneyke
"""
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

# Apply thresholding to create a binary mask
threshold_value = 0
max_value = 255
ret, binary_mask = cv2.threshold(gray_image, threshold_value, max_value, cv2.THRESH_BINARY)

# Apply the binary mask to the original image to extract the object
extracted_object = cv2.bitwise_and(image, image, mask=binary_mask)

# Display the extracted object
cv2.imshow('Extracted Object', extracted_object)
cv2.waitKey(0)
cv2.destroyAllWindows()