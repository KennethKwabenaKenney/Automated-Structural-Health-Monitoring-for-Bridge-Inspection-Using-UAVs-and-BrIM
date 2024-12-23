'''
Trying out the 2-Step Multi-Otsu Method
'''

import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageOps
from scipy.ndimage.measurements import label as cclabel
import matplotlib.colors as mcolors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu

# %%
def findMaxObj(im, num_objs=1):
    # apply connected component labeling to the binary image
    labeled_map, num_labels = cclabel(im)
    # get the number of pixels in each object
    object_sizes = np.bincount(labeled_map.flat)[1:]
    # sort the object sizes in descending order
    sorted_sizes = np.sort(object_sizes)[::-1]
    # get the labels of the largest objects
    largest_labels = np.argsort(object_sizes)[::-1][:num_objs]
    # create a binary image with the largest objects labeled as 1
    max_obj = np.zeros_like(im)
    for label in largest_labels:
        max_obj[labeled_map == label+1] = 1
    return max_obj
   

def fillHoles(im):
    im = np.pad(im, pad_width=1, mode='constant', constant_values=0)
    im = 1 - findMaxObj(1-im, 1)
    return im[1:-1,1:-1]

def otsu_threshold(image):
    # Convert image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Total number of pixels
    total_pixels = gray.shape[0] * gray.shape[1]
    
    # Variables to store optimal threshold and maximum variance
    optimal_threshold = 0
    max_variance = 0
    
    # Iterate over all possible threshold values
    for threshold in range(256):
        # Calculate the probability of foreground and background
        w0 = np.sum(hist[:threshold]) / total_pixels
        w1 = np.sum(hist[threshold:]) / total_pixels
        
        # Calculate the mean intensity of foreground and background
        u0 = np.sum(np.multiply(hist[:threshold], np.arange(threshold))) / (np.sum(hist[:threshold]) + 1e-5)
        u1 = np.sum(np.multiply(hist[threshold:], np.arange(threshold, 256))) / (np.sum(hist[threshold:]) + 1e-5)
        
        # Calculate between-class variance
        variance = w0 * w1 * (u0 - u1) ** 2
        
        # Update optimal threshold if variance is maximum
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold
    
    # Apply thresholding
    _, thresholded = cv2.threshold(gray, optimal_threshold, 255, cv2.THRESH_BINARY)
    
    return thresholded

def em_segmentation(image, num_regions):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flatten the grayscale image
    flattened = gray.flatten().reshape(-1, 1)

    # Create a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_regions)

    # Fit the GMM to the flattened image
    gmm.fit(flattened)

    # Predict the region assignments for each pixel
    pixel_assignments = gmm.predict(flattened)

    # Reshape the pixel assignments back to the original image shape
    region_map = pixel_assignments.reshape(gray.shape)

    return region_map

def otsu_threshold_without_zeros(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask for non-zero pixels
    mask = np.uint8(gray != 0)

    # Apply Otsu's method on non-zero values
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply the threshold only on non-zero pixels
    result = cv2.bitwise_and(thresholded, mask)

    return result

def remove_zeros(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Create a binary mask for non-zero pixels
    mask = gray != 0

    # Extract non-zero pixels using the mask
    image_without_zeros = image[mask]

    return image_without_zeros

def multi_otsu_segmentation1(image_path, num_classes):
    # Compute the thresholds using the multi-Otsu method
    thresholds = threshold_multiotsu(image=image_path, classes=num_classes) 

    # Apply the thresholds to obtain the segmented image
    segmented = np.digitize(image_path, bins=thresholds)

    # Return the segmented image and thresholds
    return segmented, thresholds

def multi_otsu_segmentation2(image_path, num_classes):
    # Compute the thresholds using the multi-Otsu method
    thresholds = threshold_multiotsu(image=image_path, classes=num_classes) 

    # Apply the thresholds to obtain the segmented image
    segmented = np.digitize(image_path, bins=thresholds) + 1

    # Return the segmented image and thresholds
    return segmented, thresholds

def assign_random_colors(num_colors):
    # Generate random colors
    colors = np.random.randint(0, 256, (num_colors, 3), dtype=np.uint8)
    return colors
# %%
# Load the image
root = tk.Tk()
IMAGE = filedialog.askopenfilename(title="Select Data")
root.withdraw() 

image = cv2.imread(IMAGE)
# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# %% 1st Step-Binary Otsu Method to seprate bacground and bridge
# Applying the multi-Otsu Method
first_segmented_image, computed_thresholds = multi_otsu_segmentation1(gray_image, 2)  #Number of Classes 2_binary

findMaxObj(first_segmented_image)

#fill the holes in the binary image
fill_im = fillHoles(first_segmented_image)

fill_im = fill_im.astype(np.uint8)

# Apply the binary fill image as a mask to the original image
masked_image_original = cv2.bitwise_and(image, image, mask=fill_im)

# %% Second Step Multi-Otsu Method

# Splitting the image into various channels before applying multi otsu

# Split the image into channels and trying different Combinations
blue_channel, green_channel, red_channel = cv2.split(masked_image_original)


comb1= blue_channel-(red_channel*0.85)

gray_msk_OrgImg = comb1

# Convert image to grayscale
#gray_msk_OrgImg = cv2.cvtColor(masked_image_original, cv2.COLOR_BGR2GRAY)

# Exclude zero pixel values from thresholding
mask = gray_msk_OrgImg.copy()
mask[mask == 0] = 255

# Applying the multi-Otsu Method
segmented_image, computed_thresholds = multi_otsu_segmentation2(mask, 4)  # Number of classes 4

# Map the segmented image back to the original image space
segmented_mask = np.zeros_like(gray_msk_OrgImg, dtype=np.uint8)
segmented_mask[gray_msk_OrgImg != 0] = segmented_image[gray_msk_OrgImg != 0]

# Display masked image with specified colors
unique_classes = np.unique(segmented_mask)
num_classes = len(unique_classes)

# Define specific colors for each class
class_colors = [(0, 0, 0), (255, 0, 0), (173, 216, 230), (173, 216, 230)]

# Check if the number of classes is equal to the length of class_colors
if num_classes != len(class_colors):
    # Repeat colors to match the number of classes
    class_colors = class_colors * (num_classes // len(class_colors)) + class_colors[:num_classes % len(class_colors)]

# Create a color segmented mask with specific colors for each class
color_segmented_mask = np.zeros((segmented_mask.shape[0], segmented_mask.shape[1], 3), dtype=np.uint8)
for i, class_label in enumerate(unique_classes):
    color_segmented_mask[segmented_mask == class_label] = class_colors[i]

# # Display the color segmented mask
# plt.imshow(color_segmented_mask)
# plt.show()


# Display the result
cv2.imshow('Result', masked_image_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# #Display masked image
# plt.imshow(masked_image_original)
# plt.show()
