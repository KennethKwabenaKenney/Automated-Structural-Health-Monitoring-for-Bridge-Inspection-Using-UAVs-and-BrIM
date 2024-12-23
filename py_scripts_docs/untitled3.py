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
from scipy.ndimage.measurements import label as cclabel

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
bm = binary_mask-1
# Apply the binary mask to the original image to extract the object
extracted_object = cv2.bitwise_and(image, image, mask=bm)

# Display the extracted object
cv2.imshow('Extracted Object', extracted_object)
cv2.waitKey(0)
cv2.destroyAllWindows()


bm = binary_mask-1
plt.imshow(bm)
"""
sec_rm_cc = gray_image
sec_rm_cc, __ = cclabel(sec_rm_cc, structure=np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]]))
#inverted
im = bm
unq, cnt = np.unique(im, return_counts=True)
plt.imshow(sec_rm_cc)
"""
# Load the binary mask
binary_mask = cv2.imread('binary_mask.png', cv2.IMREAD_GRAYSCALE)

# Obtain the inverse of the inverse of the binary mask
inv_mask = ~binary_mask
inv_inv_mask = ~inv_mask

# Display the binary mask and its inverse of the inverse
plt.subplot(1, 3, 1)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.subplot(1, 3, 2)
plt.imshow(inv_mask, cmap='gray')
plt.title('Inverse Mask')
plt.subplot(1, 3, 3)
plt.imshow(inv_inv_mask, cmap='gray')
plt.title('Inverse of the Inverse')
plt.show()