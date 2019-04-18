from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from scipy.signal import savgol_filter as sg
from scipy import interpolate
import os
from progressbar import progressbar_tty as ptty

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

fig = plt.figure(figsize=(7.5,7.5))
ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(111)
#ax.set_aspect(aspect='equal')
ax.set_zlim(3*floor,-1*floor)
ax.set_xlim(0,data_img.shape[1])
ax.set_ylim(-1000,data_img.shape[1]-1000)
width = 0.8

xxx = []
yyy = []
zzz = []

ddd=1
#bot
dyy,dxx = 81,81 
dd=7
zoomfactory,zoomfactorx = 1,1

print 'Plotting patterned areas...'

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

            #ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=width)
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity

#right
dyy,dxx =int(41*np.tan(np.pi*52/180)),41 
zoomfactory,zoomfactorx = 1,1
dd =20 
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
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity

#left
dyy,dxx =int(42*np.tan(np.pi*53/180)),42 
zoomfactory,zoomfactorx = 1,1
for yy in range(0,data_img.shape[0]-dyy,dyy):
    for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
        if xx>1421 or xx<332:
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
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

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
            #ax.plot_surface(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1),lw=0,cmap = 'ocean',norm= mpl.colors.Normalize(vmin=-90,vmax=1))

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial((dyy/zoomfactory,dxx/zoomfactorx), xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg_whole[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        else:
            pass
            #xopt = xstore_badtiles[(int(yy/dyy),int(xx/dxx))]


xl = np.load('thin/xleft.npy')
yl = np.load('thin/yleft.npy')
zl = np.load('thin/zleft.npy')
xr = np.load('thin/xright.npy')
yr = np.load('thin/yright.npy')
zr = np.load('thin/zright.npy')

#thinpart
print 'Interpolating thin part...'
dxx=1
offsetl = -82-2.84+1.22
offsetr = -82-1.67
if os.path.exists('xxxthin.npy'):
    xxxthin=np.load('xxxthin.npy')
    yyythin=np.load('yyythin.npy')
    zzzthin=np.load('zzzthin.npy')
    print 'Thin part loaded from existing interpolation'
else:
    xxxthin=[]
    yyythin=[]
    zzzthin=[]
    for xx in range(505,1672,dxx):
        slicing = xx 
        ylslice = [yl[i] for i in range(len(xl)) if xl[i] == slicing]
        if len(ylslice)<2:
            continue
        zlslice = [zl[i]+offsetl for i in range(len(xl)) if xl[i] == slicing]
        f = interpolate.interp1d(ylslice,zlslice,kind='linear')
        ynew = np.arange(min(ylslice),max(ylslice),10)
        znew = f(ynew)
        xxxthin.extend([xx]*len(ynew))
        yyythin.extend(ynew)
        zzzthin.extend(znew)
        #ax.plot_wireframe(X,Y,Z,rstride=int(dyy/1),cstride=int(dxx/1),colors='k',lw=0.4)
    for xx in range(2579,3703,dxx):
        slicing = xx 
        yrslice = [yr[i] for i in range(len(xr)) if xr[i] == slicing]
        if len(yrslice)<2:
            continue
        zrslice = [zr[i]+offsetr for i in range(len(xr)) if xr[i] == slicing]
        f = interpolate.interp1d(yrslice,zrslice,kind='linear')
        ynew = np.arange(min(yrslice),max(yrslice),10)
        znew = f(ynew)
        xxxthin.extend([xx]*len(ynew))
        yyythin.extend(ynew)
        zzzthin.extend(znew)
        #ax.plot_wireframe(X,Y,Z,rstride=int(dyy/1),cstride=int(dxx/1),colors='k',lw=0.4)
    print 'Thin part interpolated and saved'
    np.save('xxxthin.npy',xxxthin)
    np.save('yyythin.npy',yyythin)
    np.save('zzzthin.npy',zzzthin)
xxx.extend(xxxthin)
yyy.extend(yyythin)
zzz.extend(zzzthin)

#contact line
print 'Extracting contact line...'
x = []
y = []
xxxinterp=[]
yyyinterp=[]
zzzinterp=[]
for j in range(0,cl_img.shape[1],ddd):
#for j in range(0,2100,ddd):
    for i in range(cl_img.shape[0]-1,0,-ddd):
        if cl_img[i,j] == 1: 
            xxx.append(j)
            yyy.append(i)
            zzz.append(floor)
            xxxinterp.append(j)
            yyyinterp.append(i)
            zzzinterp.append(floor)
            x.append(j)
            y.append(i)
            break
    #ptty(j,cl_img.shape[1]/ddd,1)
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


print 'No.of points:', len(yyy)
print 'Longitudinal slicing...'
for slicing in range(0,4200,70): 
#for slicing in (1500,1600,1700): 
    yyyslice = [yyy[i] for i in range(len(xxx)) if xxx[i]==slicing]
    zzzslice = [zzz[i] for i in range(len(xxx)) if xxx[i]==slicing]
    if len(yyyslice)<4:
        continue

    zzzslice = [s for _,s in sorted(zip(yyyslice, zzzslice))]#sort zzzslice according to yyyslice
    yyyslice = sorted(yyyslice)
    duplicates = dict((i,yyyslice.count(s)) for (i,s) in enumerate(np.unique(yyyslice)) if yyyslice.count(s)>1)
    for i in duplicates:
        zzzslice[i] = np.mean(zzzslice[i:i+duplicates[i]])
        zzzslice[i+1:i+duplicates[i]] = [np.nan]*(duplicates[i]-1)
    yyyslice = np.unique(yyyslice)
    zzzslice = np.array(zzzslice)
    zzzslice = zzzslice[~np.isnan(zzzslice)]
    try:
        f = interpolate.interp1d(yyyslice,zzzslice,kind='cubic')
    except:
        continue
    #zzzslice_smooth = sg(zzzslice, window_length=5,polyorder=2)

    #ax.scatter(yyyslice,zzzslice,s=8)
    yyynew = np.arange(min(yyyslice),max(yyyslice))
    ax.plot(ys=yyynew,zs=f(yyynew),xs=len(yyynew)*[slicing],zdir='z',color="k",linewidth=0.8)
    #ax.plot(yyynew,f(yyynew))
    yyyinterp.extend(yyynew)
    zzzinterp.extend(f(yyynew))
    xxxinterp.extend(len(yyynew)*[slicing])
    ptty(slicing,3850,2)


print 'Re-processing contactline for transverse slicing...'
for i in range(0,cl_img.shape[0],ddd):
#for j in range(0,2100,ddd):
    for j in range(cl_img.shape[1]-1,int(cl_img.shape[1]*0.3),-ddd):
        if cl_img[i,j] == 1: 
            xxxinterp.append(j)
            yyyinterp.append(i)
            zzzinterp.append(floor)
            x.append(j)
            y.append(i)
            break
    for j in range(0,int(cl_img.shape[1]*0.7),ddd):
        if cl_img[i,j] == 1: 
            xxxinterp.append(j)
            yyyinterp.append(i)
            zzzinterp.append(floor)
            x.append(j)
            y.append(i)
            break

#ax.plot(x,y, 'C1',zs=floor)
print 'Transverse slicing...'
for slicing in range(300,2800,150): 
    xxxslice = [xxxinterp[i] for i in range(len(yyyinterp)) if yyyinterp[i]==slicing]
    zzzslice = [zzzinterp[i] for i in range(len(yyyinterp)) if yyyinterp[i]==slicing]
    if len(xxxslice)<4:
        continue

    zzzslice = [s for _,s in sorted(zip(xxxslice, zzzslice))]#sort zzzslice according to yyyslice
    xxxslice = sorted(xxxslice)
    duplicates = dict((i,xxxslice.count(s)) for (i,s) in enumerate(np.unique(xxxslice)) if xxxslice.count(s)>1)
    for i in duplicates:
        zzzslice[i] = np.mean(zzzslice[i:i+duplicates[i]])
        zzzslice[i+1:i+duplicates[i]] = [np.nan]*(duplicates[i]-1)
    xxxslice = list(np.unique(xxxslice))
    zzzslice = np.array(zzzslice)
    zzzslice = zzzslice[~np.isnan(zzzslice)]
    zzzslice=  list(zzzslice)
    a = xxxslice[:-1:2]+[xxxslice[-1]]
    b = zzzslice[:-1:2]+[zzzslice[-1]]
    try:
        f = interpolate.interp1d(a,b,kind='cubic')
    except Exception as e:
        print e
        continue
    ptty(slicing,max(range(300,2800,500)),1)

    #zzzslice_smooth = sg(zzzslice, window_length=5,polyorder=2)
#
    #ax.scatter(yyyslice,zzzslice,s=5)
    xxxnew = np.arange(min(xxxslice[::]),max(xxxslice[::]))
    ax.plot(xs=xxxnew,zs=f(xxxnew),ys=len(xxxnew)*[slicing],zdir='z',color="k",linewidth=.5)

plt.tight_layout()
#plt.axis('off')

#for i in range(1,240): 
#    ax.view_init(azim=60+1.5*i,elev=60)
#    plt.savefig('./movie/%d'%i+'.tif',dpi=100)

#cv2.imwrite('fitimg_whole.tif', fitimg_whole.astype('uint8'))
plt.show()
