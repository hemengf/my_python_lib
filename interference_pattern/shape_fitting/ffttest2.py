import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('ideal.tif',0)
print image.shape
nrows = np.shape(image)[0]
ncols = np.shape(image)[1]
ftimage = np.fft.fft2(image)
ftimage = np.fft.fftshift(ftimage)
logftimage = np.log(ftimage)
plt.imshow(np.abs(logftimage))

sigmax, sigmay = 10, 50
cy, cx = nrows/2, ncols/2
y = np.linspace(0, nrows, nrows)
x = np.linspace(0, ncols, ncols)
X, Y = np.meshgrid(x, y)
gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
ftimagep = ftimage * gmask
#plt.imshow(np.abs(np.log(ftimagep)))
imagep = np.fft.ifft2(ftimagep)
#plt.imshow(np.abs(imagep))




plt.show()



