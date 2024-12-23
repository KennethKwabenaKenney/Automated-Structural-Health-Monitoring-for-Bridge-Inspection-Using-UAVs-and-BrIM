import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import cv2

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

f_edge = follow_edge(canny_edges,1)
#line_join = douglas_peucker(f_edge)
#pl_dist = point_to_line_distance(line_join)

plt.imshow(f_edge)