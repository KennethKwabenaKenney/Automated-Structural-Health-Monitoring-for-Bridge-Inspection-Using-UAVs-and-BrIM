# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:23:08 2023

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

img = cv2.imread(IMAGE)

# Convert the image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Apply edge detection using Canny
edges = cv2.Canny(gray,50,150,apertureSize = 3)

# Apply Hough Transform to detect lines in the image
lines = cv2.HoughLines(edges,1,np.pi/180,200)

# Iterate through the lines detected by Hough Transform
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    # Draw the line on the image
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Display the image with detected lines
cv2.imshow('Image with detected lines', img)
cv2.waitKey(0)

# Apply probabilistic Hough Transform to detect lines in the image
lines_p = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

# Iterate through the lines detected by probabilistic Hough Transform
for line in lines_p:
    x1,y1,x2,y2 = line[0]
    # Draw the line on the image
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# Display the image with detected lines
cv2.imshow('Image with detected lines', img)
cv2.waitKey(0)

# Convert the image back to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the object from the background
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find contours in the thresholded image
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area, which is likely to be the object
cnt = max(contours, key = cv2.contourArea)

# Draw the contour on a black image
mask = np.zeros(gray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)

# Extract the object from the original image using the mask
extracted_object = cv2.bitwise_and(img,img,mask=mask)

# Display the extracted object
cv2.imshow('Extracted Object', extracted_object)
cv2.waitKey(0)
cv2.destroyAllWindows()