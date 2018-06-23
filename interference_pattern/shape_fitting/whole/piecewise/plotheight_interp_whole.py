from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from scipy import interpolate
data_img = cv2.imread('sample4.tif',0)
data_img = data_img.astype('float64') 
cl_img = cv2.imread('cl.tif',0)
cl2_img = cv2.imread('cl2.tif',0)
cl3_img = cv2.imread('cl3.tif',0)
cl3_img = cl3_img.astype('float64') 
cl3_img /= 255.
thin_img = cv2.imread('thin.tif',0)
thin_img = thin_img.astype('float64') 
thin_img /= 255.
fitimg_whole = np.copy(data_img)
xstorebot = np.load('./xoptstore_bot.npy').item()
xstoreright = np.load('./xoptstore_right.npy').item()
xstoreleft = np.load('./xoptstore_left.npy').item()
xstoretopright= np.load('./xoptstore_top_right.npy').item()
xstoretopleft= np.load('./xoptstore_top_left.npy').item()
floor = -89

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

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal')
ax.set_zlim(floor,0)
width = 0.8

xxx = []
yyy = []
zzz = []

dd=20
#bot
dyy,dxx = 81,81 
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstorebot:
            xopt = xstorebot[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy,dxx), xopt,(zoomfactory,zoomfactorx))
            if ((int(yy/dyy)+1,int(xx/dxx)) not in xstorebot) or ((int(yy/dyy)-1,int(xx/dxx)) not in xstorebot):
                pass
            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])
            height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            height[height==0] = np.nan

            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'rainbow',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy,dxx), xopt,(zoomfactory,zoomfactorx))
            height-=35
            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])
            height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            height[height==0] = np.nan
            
            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'rainbow',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
        if xx>1430:
            continue
        if (int(yy/dyy),int(xx/dxx)) in xstoreleft:
            xopt = xstoreleft[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=44

            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])
            height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            height[height==0] = np.nan

            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'rainbow',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=84

            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])
            #height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            #height[height==0] = np.nan

            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'rainbow',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
                
#topleft
dyy, dxx = 35,42
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstoretopleft:
            xopt = xstoretopleft[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=82
            #height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            #height[height==0] = np.nan

            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])

            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'rainbow',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

#dyy,dxx = 60,60
#for yy in range(0,data_img.shape[0]-dyy,dyy):
#    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
#        if thin_img[yy,xx] == 0:
#            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
#            ax.plot_wireframe(X,Y,(floor+3)*np.ones(X.shape),rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
#
#xstart,xend = 1464, 2652
#ystart,yend = 326,2823

for i in range(0,cl_img.shape[0],1):
    for j in range(0,cl_img.shape[1],1):
        if cl_img[i,j] == 255:
            xxx.append(j)
            yyy.append(i)
            zzz.append(floor)
xstart,xend = 0,data_img.shape[1] 
ystart,yend = 0,data_img.shape[0] 
xnew,ynew = np.mgrid[xstart:xend,ystart:yend]
f = interpolate.bisplrep(xxx,yyy,zzz,kx=5,ky=5)
znew  = interpolate.bisplev(xnew[:,0],ynew[0,:],f)
znew*=cl3_img.T
znew[znew == 0] =np.nan
znew[:,:300] = np.nan
x = []
y = []
ax.plot_wireframe(xnew,ynew,znew,rstride =60, cstride = 60, colors='C2',lw = width)
for j in range(cl2_img.shape[1]-1):
    for i in range(cl2_img.shape[0]-1,-1,-1):
        if cl2_img[i,j] == 0 and i>300:
            x.append(j)
            y.append(i)
            break
ax.plot(x,y, 'C2',zs=floor)
cv2.imwrite('fitimg_whole.tif', fitimg_whole.astype('uint8'))
plt.show()
