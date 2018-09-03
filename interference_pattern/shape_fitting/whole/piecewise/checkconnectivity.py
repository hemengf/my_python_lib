from scipy.ndimage import label as lb
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('cl.tif',0)
labeled_array,num =lb(img,structure=[[1,1,1],[1,1,1],[1,1,1]])
plt.imshow(labeled_array)
plt.show()
