import cv2
from scipy import ndimage as ndi
from skimage import feature
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
def equalize(img_array):
    """
    returns array with float 0-1

    """
    equalized = exposure.equalize_hist(img_array)
    return equalized 
img = cv2.imread('sample.tif',0)
img = equalize(img)
img = ndi.gaussian_filter(img,1)
edges = feature.canny(img,low_threshold=0.12,high_threshold=0.2)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
