import numpy as np
import cv2
img = cv2.imread('test.tif',0)
img = img.astype('float')
img /= 255.
#print img.sum()/(img.shape[0]*img.shape[1])
print img.sum()/len(img.flat)
