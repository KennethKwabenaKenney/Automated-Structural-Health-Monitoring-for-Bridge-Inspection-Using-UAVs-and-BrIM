# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:34:10 2023

@author: kenneyke
"""
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import cv2
import math

def point_to_point_distance(p1, p2):
    # Calculate the distance between two points using the Pythagorean formula
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# Define the line following algorithm
def follow_edge(image, start_point):
    # Initialize the list of points on the edge
    edge_points = [start_point]
    # Initialize the current point as the start point
    current_point = start_point
    # Loop until the end of the edge is reached or a junction is encountered
    while True:
        # Get the neighborhood of the current point
        neighborhood = image[current_point[0]-1:current_point[0]+2, current_point[1]-1:current_point[1]+2]
        # Find the next edge pixel in the neighborhood
        next_pixel = np.argwhere(neighborhood == 255)
        # If no edge pixel is found, the end of the edge is reached
        if len(next_pixel) == 0:
            break
        # Convert the next pixel to image coordinates
        next_point = tuple(current_point + next_pixel[0] - 1)
        # If the next point is the start point, the end of the edge is reached
        if next_point == start_point:
            break
        # If the next point is already on the edge, a junction is encountered
        if next_point in edge_points:
            break
        # Add the next point to the edge
        edge_points.append(next_point)
        # Set the next point as the current point
        current_point = next_point
    # Return the list of points on the edge
    return edge_points

# Define the line joining algorithm
def douglas_peucker(points, epsilon):
    # Find the point with the maximum distance from the line between the first and last points
    dmax = 0
    index = 0
    for i in range(1, len(points)-1):
        d = point_to_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    # If the maximum distance is greater than the threshold epsilon, recursively simplify the line
    if dmax > epsilon:
        # Simplify the line from the start point to the splitting point
        line1 = douglas_peucker(points[:index+1], epsilon)
        # Simplify the line from the splitting point to the end point
        line2 = douglas_peucker(points[index:], epsilon)
        # Combine the simplified lines
        line = line1[:-1] + line2
    # Otherwise, return the original line
    else:
        line = [points[0], points[-1]]
    # Return the simplified line
    return line

def point_to_line_distance(point, line_start, line_end):
    # Compute the distance between the point and the line between the start and end points
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / ((y2-y1)**2 + (x2-x1)**2)**0.5
    return distance

def find_epsilon(points, target_num_points):
    # Set an initial value of epsilon
    epsilon = 1.0
    # Initialize the number of points in the simplified line
    num_points = len(points)
    # Iterate until the desired number of points is reached
    while num_points > target_num_points:
        # Use the Douglas-Peucker algorithm with the current value of epsilon
        simplified_points = douglas_peucker(points, epsilon)
        # Update the number of points in the simplified line
        num_points = len(simplified_points)
        # Decrease the value of epsilon
        epsilon *= 0.9
    # Return the final value of epsilon
    return epsilon

def find_target_num_points(points):
    # Compute the total length of the input line
    total_length = 0
    for i in range(1, len(points)):
        total_length += point_to_point_distance(points[i-1], points[i])
    # Set the target number of points based on the total length
    if total_length < 1:
        target_num_points = 2
    elif total_length < 10:
        target_num_points = 4
    elif total_length < 50:
        target_num_points = 8
    elif total_length < 100:
        target_num_points = 16
    else:
        target_num_points = 32
    # Return the target number of points
    return target_num_points
#%%
# Load the image
root = tk.Tk()
IMAGE = filedialog.askopenfilename(title="Select Data")
root.withdraw() 

image = cv2.imread(IMAGE)
# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to obtain edges
canny_edges = cv2.Canny(gray_image, 50, 200)

lines = cv2.HoughLinesP(canny_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

# Apply Hough transform to detect lines in the edge image
#lines = cv2.HoughLinesP(canny_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# Select the longest line as the starting point for the edge following algorithm
longest_line = None
max_line_length = 0

for line in lines:
    x1, y1, x2, y2 = line[0]
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_length > max_line_length:
        longest_line = (x1, y1, x2, y2)
        max_line_length = line_length

# Define the starting point as the midpoint of the longest line
start_point = ((longest_line[0] + longest_line[2]) // 2, (longest_line[1] + longest_line[3]) // 2)

# Follow the edge starting from the detected starting point
edge_points = follow_edge(canny_edges, start_point)

# Find the target_num of points
target_numpt = find_target_num_points(edge_points)

# Finding the apropraite epsilon number
esp = find_epsilon(edge_points,target_numpt)

# Perform the line join from the points
douglas_peucker(edge_points,esp)

#Draw the detected starting point and the edge on the original image
cv2.circle(image, start_point, 3, (0, 0, 255), 2)
for point in edge_points:
    cv2.circle(image, point, 1, (0, 255, 0), 1)
    

#%% Result Display


#plt.imshow()