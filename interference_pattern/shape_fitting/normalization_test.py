import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from skimage import exposure

ideal_img = cv2.imread('ideal.tif', 0)
crop_img = cv2.imread('crop.tif',0)
crop_eq = exposure.equalize_hist(crop_img)
crop_eq2 = exposure.equalize_hist(crop_eq)
crop_adapteq = exposure.equalize_adapthist(crop_img, clip_limit = 0.03)
plt.imshow(crop_eq-crop_eq2)
#plt.imshow(np.concatenate((crop_eq,crop_eq2),axis=1))
plt.show()
#print np.amax(crop_eq)
#cv2.imwrite('crop_eq.tif',crop_eq)
#cv2.imwrite('crop_adapteq.tif', crop_adapteq)
#cv2.imwrite('crop_contrast_stre', crop_contrast_stre)

#density_ideal= gaussian_kde(ideal_img.flatten())
#density_crop= gaussian_kde(crop_img.flatten())
#density_ideal.covariance_factor = lambda:0.01 
#density_crop.covariance_factor = lambda:0.1 
#density_ideal._compute_covariance()
#density_crop._compute_covariance()
#x = np.linspace(0,255, 256)
hist_ideal, _ = np.histogram(ideal_img.flatten(), bins = np.amax(ideal_img))
hist_crop, _ = np.histogram(crop_img.flatten(), bins = np.amax(crop_img))
hist_crop_eq, _ = np.histogram(crop_eq.flatten(), bins = np.amax(crop_eq))
#plt.plot(ideal_img.size*density_ideal(x))
#plt.plot(hist_ideal)
#plt.plot(crop_img.size*density_crop(x)[:len(hist_crop)])
#plt.plot(hist_crop)
plt.plot(hist_crop_eq)
plt.show()
