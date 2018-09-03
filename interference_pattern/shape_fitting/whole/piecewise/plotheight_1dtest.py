from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate
from scipy.signal import savgol_filter as sg

data_img = cv2.imread('sample4.tif',0)
data_img = data_img.astype('float64') 
fitimg_whole = np.copy(data_img)
xstore = np.load('./xoptstore_bot.npy').item()
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

#dyy,dxx =int(41*np.tan(np.pi*52/180)),41 
dyy,dxx = 81,81 
zoomfactory,zoomfactorx = 1,1
fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
#ax.set_aspect('equal','box')
hslice=[]
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstore:
            xopt = xstore[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
            height = surface_polynomial((dyy,dxx), xopt,(zoomfactory,zoomfactorx))
            if int(xx/dxx) == 25:
                hslice.extend(height[:,0])


            #ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1))

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            #fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]
#cv2.imwrite('fitimg_whole.tif', fitimg_whole.astype('uint8'))
hslice_smooth=sg(hslice,window_length=81,polyorder=2)
x = range(len(hslice_smooth))[::80]
x.append(-500)
x.append(-550)
hslice_smooth=np.concatenate((hslice_smooth[::80],[-50,-50]))
#ax.plot(hslice)
f = interpolate.interp1d(np.array(x),hslice_smooth,kind='quadratic')
xnew = np.arange(-550,1500)
ax.plot(xnew,f(xnew))
#ax.plot(hslice_smooth)

plt.show()
