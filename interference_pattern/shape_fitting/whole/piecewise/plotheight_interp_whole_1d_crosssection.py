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
from matplotlib.ticker import FormatStrFormatter
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
pixratio = 12.7/4512 

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

fig,ax = plt.subplots(figsize=(7,3))
plt.subplots_adjust(bottom=0.22,left=0.2)
#ax.set_aspect(aspect='equal')
#ax.set_zlim(1.5*floor,-0.5*floor)
#ax.set_xlim(0,data_img.shape[1])
#ax.set_ylim(0,data_img.shape[0])
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
            height-=34
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
offsetl = -82-2.84+0.82#topleft
offsetr = -82-1.67
if os.path.exists('xxxthin_4paper.npy'):
    xxxthin=np.load('xxxthin_4paper.npy')
    yyythin=np.load('yyythin_4paper.npy')
    zzzthin=np.load('zzzthin_4paper.npy')
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
    np.save('xxxthin_4paper.npy',xxxthin)
    np.save('yyythin_4paper.npy',yyythin)
    np.save('zzzthin_4paper.npy',zzzthin)
    
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
#ax.plot(x,y, 'C1',zs=floor)


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
#for slicing in [1132,1748,2124]: 
for slicing, cc, cuts in zip([1132,2124],['C3','C0'],[[1010,1210-10,1260+24],[308+40,816,2344]]): 
#for slicing, cc, cuts in zip([2124],['C0'],[[308+40,816,2344]]): 
#for slicing, ls in zip([2124,3105],['C3-','k-']): 
#for slicing in range(2124,3105,50): 
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
    #ax.scatter((yyyslice),(zzzslice-floor),edgecolor='k',s=20,facecolor=cc,zorder=10)
    #ax.plot((yyyslice*pixratio),(zzzslice-floor),'r-')
    yyynew = np.arange(min(yyyslice),max(yyyslice))
    ax.plot(ys=yyynew,zs=f(yyynew),xs=len(yyynew)*[slicing],zdir='z',color="C1")
    fsmooth = sg(f(yyynew),window_length=95,polyorder=2)
    ax.plot((yyynew*pixratio),(fsmooth-floor),'k--',lw=1.7,alpha=0.5,dashes=(1.5,2))
    ax.plot((yyynew*pixratio)[:cuts[0]-yyynew[0]],(fsmooth-floor)[:cuts[0]-yyynew[0]],cc,lw=2)
    ax.plot((yyynew*pixratio)[cuts[1]-yyynew[0]:cuts[2]-yyynew[0]],(fsmooth-floor)[cuts[1]-yyynew[0]:cuts[2]-yyynew[0]],cc,lw=2)
    ax.scatter((yyynew*pixratio)[-1],0*(zzzslice-floor)[-1],edgecolor=cc,s=30,facecolor=cc,zorder=10,clip_on=False)
    #cly = (yyynew*pixratio)[-1]
    #ax.plot([cly,cly+0.5],[0,0],cc,clip_on=False,lw=2)


    yyyinterp.extend(yyynew)
    zzzinterp.extend(f(yyynew))
    xxxinterp.extend(len(yyynew)*[slicing])

"""
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
for slicing in range(300,2800,500): 
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
    #ax.plot(xs=xxxnew,zs=f(xxxnew),ys=len(xxxnew)*[slicing],zdir='z',color="C0")
"""

plt.tick_params(labelsize=18,right=True,top=True)
ax.set_xlabel(r'$z(mm)$',fontsize=24,labelpad=0)
ax.set_ylabel(r'$H(\mu m)$',fontsize=24,labelpad=0)
ax.set_yticks([0,30,60,90])
ax.set_ylim(0,90)
plt.show()
#plt.tight_layout()
#plt.axis('off')

#cv2.imwrite('fitimg_whole.tif', fitimg_whole.astype('uint8'))
