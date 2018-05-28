#!/usr/bin/env python
from __future__ import division
import sys
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from  mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from  mpl_toolkits.axes_grid1.colorbar import colorbar 
import numpy as np
import cv2
from skimage import exposure
from scipy.optimize import basinhopping
from scipy import signal


def equalize(img_array):
    """
    returns array with float 0-1

    """
    equalized = exposure.equalize_hist(img_array)
    return equalized 
	
def difference(data_img, generated_img):
    """
    both images have to be 0-1float

    """
    diff_value = np.sum((data_img-generated_img)**2)
    return diff_value

def surface_polynomial(size, coeff):
    def poly(x, y):
        poly = coeff[0]*x**2+coeff[1]*y**2+coeff[2]*x*y+coeff[3]*x+coeff[4]*y+coeff[5]
        return poly
    x = np.linspace(0,size[1]-1, size[1])
    y = np.linspace(0,size[0]-1, size[0])
    zz = poly(x[None,:],y[:,None])
    return zz

def nl(coeff, data_img):
    """
    negative likelyhood-like function; aim to minimize this
    data_img has to be 0-1float
    
    """
    height = surface_polynomial(data_img.shape, coeff)
    expected= 1+ np.cos(4*np.pi*height/0.532)
    #expected= 1+ signal.square((4*np.pi/0.532)*height)
    expected /= expected.max()#normalize to 0-1float
    #expected = equalize(expected)
    return difference(data_img, expected)

def surface_polynomial_dc(size, coeff,c):
    def poly(x, y):
        poly = coeff[0]*x**2+coeff[1]*y**2+coeff[2]*x*y+coeff[3]*x+coeff[4]*y+c/1000.
        return poly
    x = np.linspace(0,size[1]-1, size[1])
    y = np.linspace(0,size[0]-1, size[0])
    zz = poly(x[None,:],y[:, None])
    return zz

def nl_dc(coeff, data_img):
    """
    constant decoupled
    
    """
    clist =range(0,int(532/4),40)#varying c term in surface_polynomial to make stripes change at least 1 cycle
    difflist = [0]*len(clist)
    for cindx,c in enumerate(clist): 
        height = surface_polynomial_dc(data_img.shape,coeff,c)
        expected= 1+ np.cos(4*np.pi*height/0.532)
        expected /= expected.max()#normalize to 0-1float
        #expected = equalize(expected)
        difflist[cindx] = difference(data_img, expected)
    return min(difflist)/max(difflist) 

if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    import time
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    N = 50 #a,b value resolution; a, b linear term coeff
    sample_size = 60#a, b value range
    start = time.time()
    data_img = cv2.imread('sample5.tif', 0)
    fitimg = np.copy(data_img)
    xstore = {}
    dyy,dxx = 100,100
    yy,xx = 0,0
    patchysize, patchxsize = 100,100
    zoomfactory,zoomfactorx = 1,1
    data_patch = data_img[yy:yy+patchysize,xx:xx+patchxsize]
    data_patch= gaussian_filter(data_patch,sigma=0)
    data_patch = data_patch[::zoomfactory,::zoomfactorx]
    data_patch= equalize(data_patch)#float0-1

    alist = np.linspace(-sample_size,sample_size,2*N) # x direction
    blist = np.linspace(0, sample_size,N) # y direction
    #alist = np.linspace(-0.030,0.030,150) # x direction
    #blist = np.linspace(-0.030,0.030,150) # y direction
    aa, bb = np.meshgrid(alist,blist)
    nl_1storder = np.empty(aa.shape)
    for i in np.arange(alist.size):
        for j in np.arange(blist.size):
            if (j-0.5*len(blist))**2+(i)**2<=(0.*len(alist))**2:#remove central region to avoid 0,0 global min
                nl_1storder[j,i] = np.nan 
            else:
                nl_1storder[j,i] = nl([0,0,0,aa[j,i],bb[j,i],0],data_patch)
                #nl_1storder[j,i] = nl_dc([0,0,0,aa[j,i],bb[j,i]],data_patch)
            sys.stdout.write('\r%i/%i     ' % (i*blist.size+j+1,alist.size*blist.size))
            sys.stdout.flush()
    sys.stdout.write('\n')
    elapsed = time.time() - start
    print "took %.2f seconds to compute the negative likelihood" % elapsed
    index = np.unravel_index(np.nanargmin(nl_1storder), nl_1storder.shape)
    index = (alist[index[1]], blist[index[0]])
    index = np.array(index)

    initcoeff_linear= np.array([0,0,0,index[0],index[1],0])
    print initcoeff_linear

    generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, initcoeff_linear))
    generated_intensity /= generated_intensity.max()
    #generated_intenity = equalize(generated_intensity)
    plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
    plt.show() 

    nlmin = nl_1storder[~np.isnan(nl_1storder)].min()
    nlmax = nl_1storder[~np.isnan(nl_1storder)].max()
    fig = plt.figure()
    print nl_1storder.shape
    nl_1storder[np.isnan(nl_1storder)] = 0
    ax = fig.add_subplot(111)
    plt.tick_params(bottom='off',labelbottom='off',left='off',labelleft='off')
    ax.set_aspect('equal')
    print nlmin,nlmax
    im = ax.imshow(nl_1storder,cmap='RdBu',norm=mpl.colors.Normalize(vmin=nlmin,vmax=nlmax))
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right',size='3%',pad='2%')
    cbar = colorbar(im,cax = cax,ticks=[nlmin,nlmax])
    #cbar.ax.set_yticklabels(['%.1fmm/s'%lowlim,'%.1fmm/s'%78,'%.1fmm/s'%highlim])

    #fig = plt.figure()
    #plt.contour(aa, bb, nl_1storder, 100)
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(aa,bb,nl_1storder)
    #plt.ylabel("coefficient a")
    #plt.xlabel("coefficient b")
    #plt.gca().set_aspect('equal', adjustable = 'box')
    #plt.colorbar()
    plt.show()


    print 'time used', time.time()-start, 's'
    print 'finished'
