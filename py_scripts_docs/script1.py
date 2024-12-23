# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:42:13 2023

@author: kenneyke
"""
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt

# %% Functions
def otsu(im):
    hist, __ = np.histogram(im, bins=256, range=(0,255)) # FIXME: bins = 255?
    hist[0] = 0
    max_intensity_cut = int(np.max(im))
    #max_intensity_cut = np.max(im)
    sumAll = np.sum(hist*np.arange(256).astype(int))
    sumIter = 0
    q1 = 0
    N = np.count_nonzero(im)
    varMax = 0
    threshold = 0
   
    for i in range(0, max_intensity_cut):
        q1 = q1 + hist[i]
        if q1 == 0:
            continue
        q2 = N - q1
        if q2 == 0:
            break
        sumIter = sumIter + i*hist[i]
        m1 = sumIter / q1
        m2 = (sumAll - sumIter) / q2
        varBtw = q1 * q2 * (m1-m2) * (m1-m2)
        if varBtw > varMax:
            varMax = varBtw
            threshold = i
    return im > threshold


def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# %% feature list    

root = tk.Tk()
IMAGE = filedialog.askopenfilename(title="Select Data")
root.withdraw()  
 
# Load the image
img = Image.open(IMAGE)
# Convert the image to a NumPy array
img = np.array(img)


#Spliting into individual chanels for scaling
img1 = img[:,:,0]   #red Channel
img1[np.isnan(img1)]=0
img1 =scaleData(img1)
img1 = np.expand_dims(img1, axis = 2)

img2 = img[:,:,1]   #green Channel
img2[np.isnan(img2)]=0
img2 =scaleData(img2)
img2 = np.expand_dims(img2, axis = 2)

img3 = img[:,:,2]   #Blue Channel
img3[np.isnan(img3)]=0
img3 =scaleData(img3)
img3 = np.expand_dims(img3, axis = 2)


image = np.concatenate((img1, img2, img3),axis=2) #combining the individual channels into single image

# Display the image
#plt.imshow(image)

# Show the plot
#plt.show()

# Applying Otsu's thresholding
thresholded_image = otsu(img3)

# Display the thresholded image
plt.imshow(thresholded_image)

# Show the plot
plt.show()

