#!/usr/bin/env python
from __future__ import division
import sys
from scipy import interpolate
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from scipy.optimize import basinhopping

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

def surface_polynomial(size, max_variation, coeff,c):
    def poly(x, y):
        poly = max_variation*(coeff[0]*x**2+coeff[1]*y**2+coeff[2]*x*y+coeff[3]*x+coeff[4]*y)+c/1000.
        return poly
    x = np.linspace(0,size[0]-1, size[0])
    y = np.linspace(0,size[1]-1, size[1])
    zz = poly(x[None,:],y[:,None])
    return zz

def nl(coeff, max_variation, data_img):
    """
    negative likelyhood-like function; aim to minimize this
    data_img has to be 0-1float
    
    """
    clist =range(0,int(532/4),66)#varying c term in surface_polynomial to make stripes change at least 1 cycle
    difflist = [0]*len(clist)
    for cindx,c in enumerate(clist): 
        height = surface_polynomial(data_img.shape, max_variation,coeff,c)
        expected= 1+ np.cos(4*np.pi*height/0.532)
        expected /= expected.max()#normalize to 0-1float
        difflist[cindx] = difference(data_img, expected)
    return min(difflist)/max(difflist) 

if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    import time
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    N = 40 #a,b value resolution; a, b linear term coeff
    sample_size = 40#a, b value range
    start = time.time()
    max_variation = 0.001
    data_img = cv2.imread('sample.tif', 0)
    fitimg = np.copy(data_img)

    for yy in range(100,1400,100):
        for xx in range(200,700,100):#xx,yy starting upper left corner of patch
            patchysize, patchxsize = 100,100
            zoomfactory,zoomfactorx = 1,1
            data_patch = data_img[yy:yy+patchysize,xx:xx+patchxsize]
            data_patch= gaussian_filter(data_patch,sigma=0)
            data_patch = data_patch[::zoomfactory,::zoomfactorx]

            data_patch= equalize(data_patch)#float0-1
            alist = np.linspace(0,sample_size,N) # x direction
            blist = np.linspace(-sample_size, sample_size,2*N) # y direction
            aa, bb = np.meshgrid(alist,blist)
            nl_1storder = np.empty(aa.shape)

            for i in np.arange(alist.size):
                for j in np.arange(blist.size):
                    if (j-0.5*len(blist))**2+(i)**2<=(0.2*len(alist))**2:#remove central region to avoid 0,0 gloabal min
                        nl_1storder[j,i] = np.nan 
                    else:
                        nl_1storder[j,i] = nl([0,0,0,aa[j,i],bb[j,i]],max_variation,data_patch)
                    sys.stdout.write('\r%i/%i     ' % (i*blist.size+j+1,alist.size*blist.size))
                    sys.stdout.flush()
            sys.stdout.write('\n')
            elapsed = time.time() - start
            print "took %.2f seconds to compute the negative likelihood" % elapsed
            index = np.unravel_index(np.nanargmin(nl_1storder), nl_1storder.shape)
            index = (alist[index[1]], blist[index[0]])
            index = np.array(index)

            initcoeff= np.array([0,0,0,index[0],index[1]])
            print initcoeff

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, max_variation,initcoeff,0))
            generated_intensity /= generated_intensity.max()
            plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
            plt.show() 
            iternumber = 0
            itermax = 3
            while 1:
                print 'iternumber =', iternumber
                result = basinhopping(nl, initcoeff, niter = 50, T=2000, stepsize=.01, minimizer_kwargs={'method': 'Nelder-Mead', 'args': (max_variation, data_patch)}, disp=True)#, callback = lambda x, convergence, _: print('x = ', x))
                if result.fun < 0.25:
                    break
                else:
                    iternumber+=1
                    if iternumber == itermax:
                        break
                    initcoeff = result.x
            xopt = result.x
            print xopt
            clist =range(0,int(532/2),4)
            difflist = [0]*len(clist)
            for cindx,c in enumerate(clist): 
                height = surface_polynomial(data_patch.shape, max_variation,xopt,c)
                expected= 1+ np.cos(4*np.pi*height/0.532)
                expected /= expected.max()
                difflist[cindx] = difference(data_patch, expected)
            c = clist[np.argmin(difflist)]
            print [int(x) for x in difflist]
            print 'c =', c
            #fig = plt.figure()
            ##plt.contour(aa, bb, diff, 100)
            #ax = fig.add_subplot(111, projection='3d')
            #ax.plot_wireframe(aa,bb,diff)
            #plt.ylabel("coefficient a")
            #plt.xlabel("coefficient b")
            #plt.gca().set_aspect('equal', adjustable = 'box')
            #plt.colorbar()
            #plt.show()
            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, max_variation,xopt,c))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
            plt.show() 
            fitimg[yy:yy+patchysize,xx:xx+patchxsize] = 255*generated_intensity
            cv2.imwrite('fitimg.tif', fitimg.astype('uint8'))
            #cv2.imshow('', np.concatenate((generated_intensity, data_patch), axis = 1))
            #cv2.waitKey(0)
            #ax = fig.add_subplot(111, projection = '3d')
            #ax.plot_surface(xx[::10,::10], yy[::10,::10], zz[::10,::10])
            #plt.show()
    print 'time used', time.time()-start, 's'
    print 'finished'
