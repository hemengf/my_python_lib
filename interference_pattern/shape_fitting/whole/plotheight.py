from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
data_img = cv2.imread('sample5.tif')
xstore = np.load('xoptstore20180514_06_46_33.npy').item()
#xstore_badtiles=np.load('xoptstore_badtiles20180513_21_22_42.npy').item()

def surface_polynomial(size, coeff,(zoomfactory,zoomfactorx)):
    def poly(x, y):
        x*=zoomfactorx
        y*=zoomfactory
        poly = coeff[0]*x**2+coeff[1]*y**2+coeff[2]*x*y+coeff[3]*x+coeff[4]*y+coeff[5]#+coeff[6]*x**3+coeff[7]*y**3+coeff[8]*x*y**2+coeff[9]*y*x**2
        return poly
    x = np.linspace(0,size[1]-1, size[1])
    y = np.linspace(0,size[0]-1, size[0])
    zz = poly(x[None,:],y[:,None])
    return zz

dyy,dxx = 100,100
zoomfactory,zoomfactorx = 1,1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal','box')
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstore:
            xopt = xstore[(int(yy/dyy),int(xx/dxx))]
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

        X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
        height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
        ax.plot_wireframe(X,Y,height,rstride=int(dxx/5),cstride=int(dyy/5))
plt.show()
