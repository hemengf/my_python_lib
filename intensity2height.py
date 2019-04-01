from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
colorimg = cv2.imread('DSC_5311.jpg').astype(float)
#colorimg = cv2.imread('crop.tif').astype(float)
blue, green, red = cv2.split(colorimg)
#red = red*90/80
cutoff = 100
ratio = green/(red+1e-6) #prevent diverging
ratio[ratio<1] = 1 #ratio<1 not real 
lratio = np.log(ratio)

hist, bins = np.histogram(lratio.flat, bins=np.arange(0,2,0.01))
hist[np.where(hist <=cutoff)] = 0 # throw away count < cutoff
idx = np.nonzero(hist)
center = (bins[:-1] + bins[1:]) / 2

rmax = max(center[idx]) #rightmost barcenter for nonzero hist  
rmin = np.min(lratio)
lratio[lratio<rmin] = rmin
lratio[lratio>rmax] = rmax
img = (255*(lratio-rmin)/(rmax-rmin))

#width = 0.1 * (bins[1] - bins[0])
#plt.hist(lratio.flat, bins=np.arange(0,4,0.01),color='red',alpha=1)
#plt.bar(center,hist,width=width)
#plt.show()

img = img.astype('uint8')
cv2.imwrite('img.tif',img)
cv2.imwrite('green.tif', green)
cv2.imwrite('red.tif', red)
