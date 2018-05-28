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
    height = surface_polynomial(data_img.shape,coeff)
    expected= 1+ np.cos((4*np.pi/0.532)*height)
    expected /= expected.max()#normalize to 0-1float
    return difference(data_img, expected)

def accept_test(f_new,x_new,f_old,x_old):
    #return True
    if abs(x_new[3])>0.15 or abs(x_new[4])>0.15:
        return False
    else:
        return True

def callback(x,f,accept):
    #print x
    pass


if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    import time
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    N = 30 #a,b value resolution; a, b linear term coeff
    sample_size = 0.15#a, b value range
    start = time.time()
    data_img = cv2.imread('sample.tif', 0)
    fitimg = np.copy(data_img)
    xstore = {}
    dyy,dxx = 100,100
    for yy in range(0,1400,dyy):
        for xx in range(0,700,dxx):#xx,yy starting upper left corner of patch
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
                    if (j-0.5*len(blist))**2+(i)**2<=(0.2*len(alist))**2:#remove central region to avoid 0,0 global min
                        nl_1storder[j,i] = np.nan 
                    else:
                        nl_1storder[j,i] = nl([0,0,0,aa[j,i],bb[j,i],0],data_patch)
                    sys.stdout.write('\r%i/%i     ' % (i*blist.size+j+1,alist.size*blist.size))
                    sys.stdout.flush()
            sys.stdout.write('\n')
            elapsed = time.time() - start
            print "took %.2f seconds to compute the negative likelihood" % elapsed
            index = np.unravel_index(np.nanargmin(nl_1storder), nl_1storder.shape)
            index = (alist[index[1]], blist[index[0]])
            index = np.array(index)

            initcoeff_linear= np.array([0,0,0,index[0],index[1],0])
            #print initcoeff_linear

            initcoeff_extendlist = []
            if (int(yy/dyy)-1,int(xx/dxx)) in xstore:
                up = xstore[(int(yy/dyy)-1,int(xx/dxx))]
                initcoeff_extendlist.append(np.array([up[0],up[1],up[2],up[2]*dyy+up[3],2*up[1]*dyy+up[4],up[1]*dyy*dyy+up[4]*dyy+up[5]]))
            if (int(yy/dyy),int(xx/dxx)-1) in xstore:
                left = xstore[(int(yy/dyy),int(xx/dxx)-1)]
                initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx+left[3],left[2]*dxx+left[4],left[0]*dxx*dxx+left[3]*dxx+left[5]]))
            else:
                print 'no calculated neighbours found...'
            if len(initcoeff_extendlist) > 0:
                initcoeff_extend = np.mean(initcoeff_extendlist,axis=0)
            else:
                initcoeff_extend = initcoeff_linear

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, initcoeff_linear))
            generated_intensity /= generated_intensity.max()
            plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
            #plt.show() 
            #initcoeff_extend = np.array([0,0,0,0,0,0])
            iternumber = 0
            while 1:
                print 'iternumber =', iternumber,'for',xx,yy
                result = basinhopping(nl, initcoeff_extend, niter = 100, T=100, stepsize=0.0001, interval=20,accept_test=accept_test,minimizer_kwargs={'method': 'Nelder-Mead', 'args': (data_patch)}, disp=True, callback=callback)
                if result.fun < 520:
                    break
                else:
                    initcoeff_extend = result.x
                    iternumber+=1
                    if iternumber == 2:
                        initcoeff_extend = initcoeff_linear
                        print 'using linear coefficients'
                    if iternumber == 2:
                        break
            xopt = result.x
            xstore[(int(yy/100),int(xx/100))]=xopt

            #print xopt
            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, xopt))
            generated_intensity /= generated_intensity.max()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            #plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
            #plt.show() 
            fitimg[yy:yy+patchysize,xx:xx+patchxsize] = 255*generated_intensity
            cv2.imwrite('fitimg.tif', fitimg.astype('uint8'))
    print 'time used', time.time()-start, 's'
    print 'finished'
