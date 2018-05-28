import cv2
import numpy as np
from skimage import transform as tf
import matplotlib.pyplot as plt
img = cv2.imread('sample6.tif',0)
pointset1 = np.genfromtxt('pointset1.csv', delimiter=',', names=True)
pointset2 = np.genfromtxt('pointset2.csv', delimiter=',', names=True)
pointset1 = np.vstack((pointset1['BX'],pointset1['BY'])).T
pointset2 = np.vstack((pointset2['BX'],pointset2['BY'])).T
tform = tf.PiecewiseAffineTransform()
tform.estimate(pointset1, pointset2) # pointset2 will be warped
warped = 255*tf.warp(img, tform)
warped =  warped.astype(np.uint8)
plt.imshow(warped)
plt.show()
