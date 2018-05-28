from scipy import fftpack
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('ideal.tif',0)
absfft2 = np.abs(fftpack.fft2(img))[2:-2,2:-2]
absfft2 /= absfft2.max()
print absfft2.max()
plt.imshow(absfft2)
plt.show()
