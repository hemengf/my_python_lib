from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from scipy import interpolate
import os

data_img = cv2.imread('sample4.tif',0)
data_img = data_img.astype('float64') 
cl_img = cv2.imread('cl.tif',0)
cl2_img = cv2.imread('cl2_larger.tif',0)
cl3_img = cv2.imread('cl3.tif',0)
edge_img = cv2.imread('cl_edge.tif',0)
thin_img = cv2.imread('thin.tif',0)

cl_img = cl_img.astype('float64') 
cl_img /= 255.

cl2_img = cl2_img.astype('float64') 
cl2_img /= 255.

cl3_img = cl3_img.astype('float64') 
cl3_img /= 255.

edge_img = edge_img.astype('float64') 
edge_img /= 255.

thin_img = thin_img.astype('float64') 
thin_img /= 255.

fitimg_whole = np.copy(data_img)

xstorebot = np.load('./xoptstore_bot.npy').item()
xstoreright = np.load('./xoptstore_right.npy').item()
xstoreleft = np.load('./xoptstore_left.npy').item()
xstoretopright= np.load('./xoptstore_top_right.npy').item()
xstoretopleft= np.load('./xoptstore_top_left.npy').item()

floor = -86

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
ax.set_aspect(adjustable='datalim',aspect='equal')
ax.set_zlim(floor,0)
width = 0.8

xxx = []
yyy = []
zzz = []

ddd=1
#bot
dyy,dxx = 81,81 
dd=15
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
            #height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            #height[height==0] = np.nan

            ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
dd = 5
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if xx > 3850:
            continue
        if (int(yy/dyy),int(xx/dxx)) in xstoreright:
            xopt = xstoreright[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy,dxx), xopt,(zoomfactory,zoomfactorx))
            height-=35
            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])
            #height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            #height[height==0] = np.nan
            
            #ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
        if xx>1430 or xx<332:
            continue
        if (int(yy/dyy),int(xx/dxx)) in xstoreleft:
            xopt = xstoreleft[(int(yy/dyy),int(xx/dxx))]
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            height = surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx))
            height-=44

            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])
            #height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            #height[height==0] = np.nan

            #ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
            height-=82

            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])
            #height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            #height[height==0] = np.nan

            #ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
            height-=80.3
            #height*= 1-cl3_img[yy:yy+dyy,xx:xx+dxx]
            #height[height==0] = np.nan

            xxx+=list(X.flat[::dd])
            yyy+=list(Y.flat[::dd])
            zzz+=list(height.flat[::dd])

            #ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]

dyy,dxx =60,60 
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if thin_img[yy,xx] == 0:
            xxx.append(xx)
            yyy.append(yy)
            zzz.append(floor+3)
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(yy,yy+dyy,zoomfactory))
            Z = (floor+3)*np.ones(X.shape)
            Z*= 1-thin_img[yy:yy+dyy,xx:xx+dxx]
            Z[Z==0] = np.nan
            ax.plot_wireframe(X,Y,Z,rstride=int(dyy/1),cstride=int(dxx/1),colors='k',lw=0.4)


if os.path.exists('./znew.npy'):
    xstart,xend = 0,data_img.shape[1] 
    ystart,yend = 0,data_img.shape[0] 
    xnew,ynew = np.mgrid[xstart:xend,ystart:yend]
    znew = np.load('znew.npy')
    znew[znew<floor] = np.nan
    znew*=(thin_img).T
    znew*=(cl2_img).T
    znew[znew == 0] =np.nan
    znew[:,:300] = np.nan
    ax.plot_wireframe(xnew,ynew,znew,rstride =60, cstride = 60, colors='k',lw = 0.4)
    #ax.plot_surface(xnew,ynew,znew,rstride=40,cstride=40,lw=0,cmap = 'RdBu',norm= mpl.colors.Normalize(vmin=-90,vmax=1))
else:
    for i in range(0,cl_img.shape[0],ddd):
        for j in range(0,cl_img.shape[1],ddd):
            if cl_img[i,j] == 1: 
                xxx.append(j)
                yyy.append(i)
                zzz.append(floor)
    xstart,xend = 0,data_img.shape[1] 
    ystart,yend = 0,data_img.shape[0] 
    xnew,ynew = np.mgrid[xstart:xend,ystart:yend]

    print 'interpolating'
    f = interpolate.bisplrep(xxx,yyy,zzz,kx=5,ky=3)
    print 'finished'
    znew  = interpolate.bisplev(xnew[:,0],ynew[0,:],f)
    znew[znew<floor] =np.nan
    znew*=(thin_img).T
    znew*=(cl2_img).T
    znew[znew == 0] =np.nan
    znew[:,:300] = np.nan
    np.save('znew.npy',znew)
    ax.plot_wireframe(xnew,ynew,znew,rstride =60, cstride = 60, colors='k',lw = 0.4)

x = []
y = []
for j in range(0,cl_img.shape[1]-1,5):
    for i in range(cl_img.shape[0]-1,-1,-5):
        if cl_img[i,j] == 1 and i>200:
            x.append(j)
            y.append(i)
            break
ax.plot(x,y, 'k',zs=floor)

#x_edge=[]
#y_edge=[]
#z_edge=[]
#for i in range(0,edge_img.shape[0],2):
#    for j in range(0,edge_img.shape[1],2):
#        if edge_img[i,j] == 1:
#            x_edge.append(j)
#            y_edge.append(i)
#            z_edge.append(znew[j,i])
#ax.scatter(x_edge,y_edge,z_edge,c='k',s=0.01)

#ax.view_init(azim=-122,elev=75)
ax.view_init(azim=128,elev=75)
#plt.axis('off')
plt.tight_layout()
#cv2.imwrite('fitimg_whole.tif', fitimg_whole.astype('uint8'))
plt.show()
