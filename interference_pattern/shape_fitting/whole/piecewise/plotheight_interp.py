from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate
data_img = cv2.imread('sample4.tif',0)
data_img = data_img.astype('float64') 
xstore = np.load('./xoptstore_bot.npy').item()
xstorebot = np.load('./xoptstore_bot.npy').item()
xstoreright = np.load('./xoptstore_right.npy').item()
xstoreleft = np.load('./xoptstore_left.npy').item()
xstoretopright= np.load('./xoptstore_top_right.npy').item()
xstoretopleft= np.load('./xoptstore_top_left.npy').item()
cl_img = cv2.imread('cl.tif',0)
cl2_img = cv2.imread('mask_bot_v2.tif',0)
fitimg_whole = np.copy(data_img)

cl2_img = cl2_img.astype('float64') 
cl2_img /= 255.

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
floor = -89
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect(adjustable='datalim',aspect='equal')
ax.set_zlim(floor,0)
width = 0.8
dd=80
ddd=20

xxx = []
yyy = []
zzz = []

for i in range(0,cl_img.shape[0],ddd):
    for j in range(0,cl_img.shape[1],ddd):
        if cl_img[i,j] == 255:
            xxx.append(j)
            yyy.append(i)
            zzz.append(floor)
#bot
dyy,dxx = 81,81 
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if (int(yy/dyy),int(xx/dxx)) in xstore:
            xopt = xstore[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy,dxx), xopt,(zoomfactory,zoomfactorx))
            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])

            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]
 
#xstart,xend = 1698,1942
#ystart,yend = 1726,2323
xstart,xend = 0,data_img.shape[1] 
ystart,yend = 0,data_img.shape[0] 
print 'interpolating'
f = interpolate.interp2d(xxx,yyy,zzz,kind='quintic')
print 'finish'
XX,YY = np.meshgrid(range(xstart,xend),range(ystart,yend))
ZZ = f(range(xstart,xend),range(ystart,yend))
ZZ*=cl2_img[ystart:yend,xstart:xend]
ZZ[ZZ == 0] =np.nan
ZZ[:,:300] = np.nan
ax.plot_wireframe(XX,YY,ZZ,rstride =80, cstride = 80, colors='k',lw=0.4)
#ax.contour3D(XX,YY,ZZ,50,cmap='binary')
cv2.imwrite('fitimg_whole.tif', fitimg_whole.astype('uint8'))
plt.show()
