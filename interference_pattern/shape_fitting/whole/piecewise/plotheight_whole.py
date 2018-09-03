from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
data_img = cv2.imread('sample4.tif',0)
fitimg_whole = np.copy(data_img)
xstorebot = np.load('./xoptstore_bot.npy').item()
xstoreright = np.load('./xoptstore_right.npy').item()
xstoreleft = np.load('./xoptstore_left.npy').item()
xstoretopright= np.load('./xoptstore_top_right.npy').item()
xstoretopleft= np.load('./xoptstore_top_left.npy').item()
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

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal','box')

#bot
width=0.8
dyy,dxx = 81,81
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstorebot:
            xopt = xstorebot[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            ax.plot_wireframe(X,Y,height,rstride=int(dyy/2),cstride=int(dxx/2),lw=width)

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

#right
dyy,dxx =int(41*np.tan(np.pi*52/180)),41 
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstoreright:
            xopt = xstoreright[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=35
            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

#left
dyy,dxx =int(42*np.tan(np.pi*53/180)),42 
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        #if xx>1430:
            #continue
        if (int(yy/dyy),int(xx/dxx)) in xstoreleft:
            xopt = xstoreleft[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=44
            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

#topright
dyy, dxx = 35,42
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstoretopright:
            xopt = xstoretopright[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=84
            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

#topleft
dyy, dxx = 35,42
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstoretopleft:
            xopt = xstoretopleft[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=82
            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

#cv2.imwrite('fitimg_whole.tif', fitimg_whole.astype('uint8'))
plt.show()
