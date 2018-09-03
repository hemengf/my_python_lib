from __future__ import division
import numpy as np
import cv2
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

left0_img = cv2.imread('left0.tif',0)
left1_img = cv2.imread('left1.tif',0)
left2_img = cv2.imread('left2.tif',0)
left3_img = cv2.imread('left3.tif',0)
left4_img = cv2.imread('left4.tif',0)
leftflat_img = cv2.imread('leftflat.tif',0)

right0_img = cv2.imread('right0.tif',0)
right1_img = cv2.imread('right1.tif',0)
right2_img = cv2.imread('right2.tif',0)
right3_img = cv2.imread('right3.tif',0)
right4_img = cv2.imread('right4.tif',0)

xl=[]
yl=[]
zl=[]
xr=[]
yr=[]
zr=[]
dd=1
offsetl = 0
offsetr = 0 
for i in range(252,1046,dd):
    for j in range(505,1672,dd):
        if left0_img[i,j] == 255:
            xl.append(j)
            yl.append(i)
            zl.append(0+offsetl)
        if left1_img[i,j] == 255:
            xl.append(j)
            yl.append(i)
            zl.append(1*0.532/2+offsetl)
        if left2_img[i,j] == 255:
            xl.append(j)
            yl.append(i)
            zl.append(2*0.532/2+offsetl)
        if left3_img[i,j] == 255:
            xl.append(j)
            yl.append(i)
            zl.append(3*0.532/2+offsetl)
        if left4_img[i,j] == 255:
            xl.append(j)
            yl.append(i)
            zl.append(4*0.532/2+offsetl)
        #if leftflat_img[i,j] == 255:
        #    xl.append(j)
        #    yl.append(i)
        #    zl.append(2.5*0.532/2)
for i in range(272,1012,dd):
    for j in range(2579,3703,dd):
        if right0_img[i,j] == 255:
            xr.append(j)
            yr.append(i)
            zr.append(0+offsetr)
        if right1_img[i,j] == 255:
            xr.append(j)
            yr.append(i)
            zr.append(1*0.532/2+offsetr)
        if right2_img[i,j] == 255:
            xr.append(j)
            yr.append(i)
            zr.append(2*0.532/2+offsetr)
        if right3_img[i,j] == 255:
            xr.append(j)
            yr.append(i)
            zr.append(3*0.532/2+offsetr)
        if right4_img[i,j] == 255:
            xr.append(j)
            yr.append(i)
            zr.append(4*0.532/2+offsetr)

np.save('xleft.npy',xl)
np.save('yleft.npy',yl)
np.save('zleft.npy',zl)
np.save('xright.npy',xr)
np.save('yright.npy',yr)
np.save('zright.npy',zr)
"""
slicing = 1128
yslice = [y[i] for i in range(len(x)) if x[i] == slicing]
zslice = [z[i] for i in range(len(x)) if x[i] == slicing]
f = interpolate.interp1d(yslice,zslice,kind='linear')
xnew = np.arange(min(x),max(x))
ynew = np.arange(min(yslice),max(yslice))
znew = f(ynew)
#XX,YY = np.meshgrid(xnew,ynew)
#fig = plt.figure(figsize=(7,7))
#ax = fig.add_subplot(111,projection='3d')
#ax.set_zlim(0,1000)
#ax.plot_wireframe(XX,YY,znew)
#ax.scatter(x,y,z)
plt.plot(ynew,znew)
plt.scatter(yslice, zslice)
plt.show()
"""
