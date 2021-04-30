import numpy as np
import matplotlib.pyplot as plt
import cv2
# from google.colab import files as FILE
import os, requests


# Read in the image
image = cv2.imread('/home/ale/Documents/GitHub/FaceDataset/BacteriaDataset/images/196.png')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# Convert the image to gray Scale
grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)




# The PCA module in Sklearn is the most known implementation of PCA.
from sklearn.decomposition import PCA

# First, let's define the number of principal components
n_comp = 50

# Initialize PCA
pca = PCA(n_components=n_comp)

# Standardize the data, so all instances have 0 as the center
pca.fit(grayscale) 

# Find the (n_comp) number of principal components and remove the less important 
# Theres's also another function that joins fit and tranform: pca.Fit_transform()
principal_components = pca.transform(grayscale) 

# PCA.transform also finds the explained_variance_ratio_ , 
# which shows the % of variance explained by each component
# print("explained_variance_ratio_:",pca.explained_variance_ratio_)

# Since PCA reduces the number of columns, we will need to transform the results 
# to the original space to display the compressed image
temp = pca.inverse_transform(principal_components) 



# Implement canny edge detection
temp = np.uint8(temp)
# Implement canny edge detection
canny = cv2.Canny(temp, 50, 150)

# Implement image threshold 
ret, thresh = cv2.threshold(temp,100,180,cv2.THRESH_BINARY_INV)



s_1 = np.uint8([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0], 
                   [0, 0, 1, 0, 0], 
                    ])
s_2 = np.ones((5,5),np.uint8)



# Reads in a binary image
copy6 = np.copy(thresh)

# Opening the image
closing = cv2.morphologyEx(copy6, cv2.MORPH_CLOSE, s_1)

# Opening the image
closing2 = cv2.morphologyEx(copy6, cv2.MORPH_CLOSE, s_2)



# Threshold.
th, im_th = cv2.threshold(closing, 50,150, cv2.THRESH_BINARY_INV);
 
# Copy the thresholded image.
im_floodfill = im_th.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv



# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(im_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
image = cv2.drawContours(im_out, contours, -1, (0, 255, 0), 3)


print("Si existe más de 25 % se toma en cuenta la existencia de bacterias")
print("Bacterias: ", len(contours))
if np.mean(im_out)>25:
  print("Existe bacterias, el porcentaje: ", np.mean(im_out))
else:
  print("No existe muchas bacterias, el porcentaje: ", np.mean(im_out))