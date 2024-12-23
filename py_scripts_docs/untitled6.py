import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import cv2
import keras
from keras.metrics import MeanIoU
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

def multi_otsu_segmentation(image_path, num_classes):
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

# Apply Canny edge detection to obtain edges
canny_edges = cv2.Canny(gray_image, 50, 200)

# Set edge pixels to 1 and non-edge pixels to 0
canny_edges[canny_edges != 0] = 1

# Invert the binary image
canny_edges = 1 - canny_edges

# Apply connected component labeling to the binary image
labeled_map, num_labels = cclabel(canny_edges)    # plt.imshow(labeled_map, cmap='jet')

unq, cnt = np.unique(labeled_map, return_counts=True)

segment_map = labeled_map.copy() # pick up the largest segment here, but it might not work well on other images. -> should be automated 
segment_map[segment_map != 1] = 0  #  plt.imshow(segment_map, cmap='jet')
# based on the RGB or intensity values, maybe you can separate bridge segments from others 
# Otsu method can be a solution 

# Set all labels except label with value 1 to zero
# labeled_map = np.where(labeled_map == 1, labeled_map, 0)

# Select label 1 in the labeled map
# label_1 = (labeled_map == 1)

# Use label as mask to cut out original image
# masked_image = np.zeros_like(image)
# masked_image[segment_map==1,:] = image[segment_map==1,:]

masked_image = image.copy()
masked_image[segment_map!=1,0] = 0
masked_image[segment_map!=1,1] = 0
masked_image[segment_map!=1,2] = 0

# Display masked image
#plt.imshow(masked_image)
#plt.show()

region_img = masked_image

#Region Growing for the masked data for extra Refinement
# Convert the image to grayscale
gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)

# Set the seed point for region growing
seed = (381, 254)

# Create a mask with zeros
mask = np.zeros((gray.shape[0]+2, gray.shape[1]+2), np.uint8)

# Set up the region growing parameters
neighborhood = 4
flags = neighborhood + cv2.FLOODFILL_FIXED_RANGE

# Run the region growing algorithm with a threshold of 50
threshold = 50
cv2.floodFill(gray, mask, seed, (255, 255, 255), (threshold,) * 3, (threshold,) * 3, flags)

# Extract the object from the image using the mask
object = cv2.bitwise_and(region_img, region_img, mask=mask[1:-1, 1:-1])

# Convert to grayscale
gray_object = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
thresh = 1
binary_img = cv2.threshold(gray_object, thresh, 1, cv2.THRESH_BINARY)[1]

findMaxObj(binary_img)

#fill the holes in the binary image
fill_im = fillHoles(binary_img)

# Apply the binary fill image as a mask to the original image
masked_image_original = cv2.bitwise_and(image, image, mask=fill_im)

# %% Binary Otsu MEthod
# Convert image to grayscale
#gray_msk_OrgImg = cv2.cvtColor(masked_image_original, cv2.COLOR_BGR2GRAY)

#Remove zeros
#ga = gray_msk_OrgImg[gray_msk_OrgImg>0]

# Applying the binary Otsu Method
#newImg = otsu_threshold(ga)

# %% Multi-Otsu Method

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
segmented_image, computed_thresholds = multi_otsu_segmentation(mask, 4)  # Number of classes 4

# Map the segmented image back to the original image space
segmented_mask = np.zeros_like(gray_msk_OrgImg, dtype=np.uint8)
segmented_mask[gray_msk_OrgImg != 0] = segmented_image[gray_msk_OrgImg != 0]

# Display masked image with random colors
unique_classes = np.unique(segmented_mask)
num_classes = len(unique_classes)

# # Assign random colors to each class
# colors = assign_random_colors(num_classes)

# # Create a color image with random colors for each class
# color_segmented_mask = np.zeros((segmented_mask.shape[0], segmented_mask.shape[1], 3), dtype=np.uint8)
# for i, class_label in enumerate(unique_classes):
#     color_segmented_mask[segmented_mask == class_label] = colors[i]

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

# #Display the result
# cv2.imshow('Result', masked_image_original)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #Display masked image
# plt.imshow(gray)
# plt.show()


# %% Perform metric quantification of predicted results

## Converting the 4-class labeled image into 2-classes
# Define the labels for each class
background_label = 0
bridge_label = 1
class1_label = 2
class2_label = 3

# Convert the segmented mask to a binary mask with two classes
binary_mask = np.zeros_like(segmented_mask, dtype=np.uint8)
binary_mask[segmented_mask == background_label] = 2  # Background pixels
binary_mask[segmented_mask == bridge_label] = 1  # Bridge pixels
binary_mask[segmented_mask == class1_label] = 2  # Class 1 pixels (set to bridge)
binary_mask[segmented_mask == class2_label] = 2  # Class 2 pixels (set to bridge)

# # Display the binary mask
# plt.imshow(binary_mask, cmap='gray')
# plt.show()

# Setting the ground truth data
truth = binary_mask
#truth = Image.open(r'D:\My_Research\Data\2_ground_truth.PNG')
prediction = binary_mask

# Convert truth and prediction to numpy arrays
truth = np.array(truth)
prediction = np.array(prediction)

#Accuracies
#Overall Accuracy
from sklearn import metrics
print('Accuracy =',metrics.accuracy_score(truth,prediction))

# Using mean IoU
num_classes = 1
IOU_KERAS = MeanIoU(num_classes=num_classes)
IOU_KERAS.update_state(truth,prediction)
print("Mean IoU =",IOU_KERAS.result().numpy())
# %%
# Convert the segmented image back to 8-bit
#segmented_image = (segmented_image * 255).astype(np.uint8)
# Remove zero values from the image
#image_without_zeros = remove_zeros(masked_image_original)



# Apply Otsu's method while excluding zeros
#result = otsu_threshold_without_zeros(masked_image_original)
# %%
# Apply Otsu thresholding
#result = otsu_threshold(masked_image_original)

# Set the number of desired regions for EM segmentation
#num_regions = 10

# Apply EM segmentation
#result = em_segmentation(masked_image_original, num_regions)

# Generate random colors for each region
#colors = np.random.randint(0, 256, size=(num_regions, 3), dtype=np.uint8)

# Assign colors to the regions
#result_colored = colors[result]

# %%
# Convert image to grayscale
#gray_image_hist = cv2.cvtColor(masked_image_original, cv2.COLOR_BGR2GRAY)

# Calculate the histogram
#histogram = cv2.calcHist([gray_image_hist], [0], None, [256], [0, 256])

# Flatten the histogram array
#histogram = histogram.flatten()

# Exclude zero pixel values from the histogram
#histogram_nonzero = histogram[histogram.nonzero()]

# Plot the histogram
#plt.figure()
#plt.plot(histogram, color='black')
#plt.xlabel('Pixel Intensity')
#plt.ylabel('Frequency')
#plt.title('Histogram of Image')
#plt.xlim([200, 256])  # Set the x-axis range from 0 to 256 (intensity range)
# Reduce the y-axis range to fit the new histogram
#plt.ylim([0, 30000])
#plt.ylim([0, max(histogram_nonzero)]) 
#plt.show()



# Display the result
#cv2.imshow('masked_image_original', masked_image_original)
#cv2.imshow('Result', segmented_image)
#cv2.imshow('gray_image_hist', gray_image_hist)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Display masked image
#plt.imshow(newImg)
#plt.show()

'''
###### From my Prof for out next meeting
#--- Otsu nethod
#--- EM segmentation
#---- k-means
b = b[b>0]
b is 1d array
c is also 1d array
input b and c have the same size
to map the classification result back to original 2d image
b[b>0] = c
b is 2d array
[b>0]
b = b[b>0]
y = b[b>0]
b[b>0] = classification

'''
'''

(1) from skimage.filters import threshold_multiotsu
(2) plot the intensity values using the histogram
(3) find the distribution corresponding to the damaged area
(4) from the distribution, find a threshold corresponding to the damaged area

*a = a[a>0]*
also try otsu method but with nulti-class so that you dont focus only on binary i.e multi-otsu method
try area base filtering 

#plt.imshow(fill_im)
#plt.show()
'''
'''
# Use the new label as a mask to cut out the original image
masked_image = np.where(labeled_map == 1, gray_image, 0)

# Convert grayscale image to color
color_image = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)

# Display masked color image
plt.imshow(color_image)
plt.show()

# Check the shape of the object image
if len(object.shape) == 2:
    # Object image is grayscale, so convert it to BGR
    object = cv2.cvtColor(object, cv2.COLOR_GRAY2BGR)
elif object.shape[2] == 4:
    # Object image has an alpha channel, so remove it
    object = cv2.cvtColor(object, cv2.COLOR_BGRA2BGR)


# Define a list of colors for the discrete colormap
colors = ['black', 'blue']

# Create a discrete colormap using the ListedColormap class
cmap = mcolors.ListedColormap(colors)

# Display the labeled connected components with the discrete colormap
plt.imshow(labeled_map, cmap=cmap)
plt.colorbar()
plt.show()

#-- assignement #1

colormap = np.array([])  (n by 4) 0: R, 1: G, 2:B, 3: assign 1 
plug colormap into imshow


# Define a list of colors for the discrete colormap
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'violet', 'black', 'gray']

# Create a discrete colormap using the ListedColormap class
cmap = mcolors.ListedColormap(colors)

# Display the labeled connected components with the discrete colormap
plt.imshow(labeled_map, cmap=cmap)
plt.colorbar()
plt.show()


#Display the labeled connected components
#plt.imshow(labeled_map)
#plt.colorbar()
#plt.show()

#plt.imshow(canny_edges)
'''

'''
1. Try out processing the otsu method on single/ individual bands nad or combinations and combination ratios
2. Try the otsu method automatically without any manual classes 
3. Do literature review on our current state
4. Find more data from the internet and reference them (for cracks and corrosion)
'''