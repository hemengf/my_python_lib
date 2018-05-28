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
from scipy import fftpack
from scipy import signal

def equalize(img_array):
    """
    returns array with float 0-1

    """
    equalized = exposure.equalize_hist(img_array)
    #equalized = img_array/img_array.max()
    return equalized 

def difference(data_img, generated_img):
    """
    both images have to be 0-1float

    """
    diff_value = np.sum((data_img-generated_img)**2)
    return diff_value

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

def nl(coeff, data_img,(zoomfactory,zoomfactorx)):
    """
    negative likelyhood-like function; aim to minimize this
    data_img has to be 0-1float
    
    """
    height = surface_polynomial(data_img.shape,coeff,(zoomfactory,zoomfactorx))
    expected= 1+ np.cos((4*np.pi/0.532)*height)
    expected /= expected.max()#normalize to 0-1float
    #expected = equalize(expected)
    return difference(data_img, expected)

def accept_test(f_new,x_new,f_old,x_old):
    return True
    if abs(x_new[3])>0.05 or abs(x_new[4])>0.05:
        return False
    else:
        return True

def callback(x,f,accept):
    #print x[3],x[4],f,accept
    pass


if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    import time
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from time import localtime, strftime

    N = 50 #a,b value resolution; a, b linear term coeff
    sample_size = 0.2#a, b value range
    start = time.time()
    data_img = cv2.imread('sample7.tif', 0)
    abquadrant = 3
    fitimg = np.copy(data_img)
    xstore = {}
    xstore_badtiles = {}
    hstore_upperright = {}
    hstore_lowerright = {}
    hstore_lowerleft = {}
    threshold = 300
    dyy,dxx = 44,60
    zoomfactory,zoomfactorx = 1,1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for yy in range(0,data_img.shape[0]-dyy,dyy):
        for xx in range(0,data_img.shape[1]-dxx,dxx):#xx,yy starting upper left corner of patch
    #for yy in range(0,1,dyy):
    #    for xx in range(0,1,dxx):#xx,yy starting upper left corner of patch
            print 'processing', (int(yy/dyy),int(xx/dxx))
            data_patch = data_img[yy:yy+dyy,xx:xx+dxx]
            data_patch= gaussian_filter(data_patch,sigma=0)
            data_patch = data_patch[::zoomfactory,::zoomfactorx]

            data_patch= equalize(data_patch)#float0-1

            initcoeff_extendlist = []
            if (int(yy/dyy)-1,int(xx/dxx)) in xstore:
                print 'found up'
                up = xstore[(int(yy/dyy)-1,int(xx/dxx))]
                initcoeff_extendlist.append(np.array([up[0],up[1],up[2],up[2]*dyy+up[3],2*up[1]*dyy+up[4],up[1]*dyy*dyy+up[4]*dyy+up[5]]))
            if (int(yy/dyy),int(xx/dxx)-1) in xstore:
                print 'found left'
                left = xstore[(int(yy/dyy),int(xx/dxx)-1)]
                initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx+left[3],left[2]*dxx+left[4],left[0]*dxx*dxx+left[3]*dxx+left[5]]))
            if len(initcoeff_extendlist) > 0:
                initcoeff_extend = np.mean(initcoeff_extendlist,axis=0)
                initcoeff = initcoeff_extend
            else:
                if abquadrant == 1:
                    alist = np.linspace(0, sample_size, N) # x direction
                    blist = np.linspace(0, sample_size, N) # y direction
                if abquadrant == 2:
                    alist = np.linspace(-sample_size, 0, N) # x direction
                    blist = np.linspace(0, sample_size, N) # y direction
                if abquadrant == 3:
                    alist = np.linspace(-sample_size, 0, N) # x direction
                    blist = np.linspace(-sample_size, 0, N) # y direction
                if abquadrant == 4:
                    alist = np.linspace(0, sample_size, N) # x direction
                    blist = np.linspace(-sample_size, 0, N) # y direction
                aa, bb = np.meshgrid(alist,blist)
                nl_1storder = np.empty(aa.shape)
                for i in np.arange(alist.size):
                    for j in np.arange(blist.size):
                        if (j-0.5*len(blist))**2+(i)**2<=(0.1*len(alist))**2:#remove central region to avoid 0,0 global min
                            nl_1storder[j,i] = np.nan 
                        else:
                            nl_1storder[j,i] = nl([0,0,0,aa[j,i],bb[j,i],0],data_patch,(zoomfactory,zoomfactorx))
                        sys.stdout.write('\r%i/%i     ' % (i*blist.size+j+1,alist.size*blist.size))
                        sys.stdout.flush()
                sys.stdout.write('\n')
                elapsed = time.time() - start
                index = np.unravel_index(np.nanargmin(nl_1storder), nl_1storder.shape)
                index = (alist[index[1]], blist[index[0]])
                index = np.array(index)

                initcoeff_linear= np.array([0,0,0,index[0],index[1],0])
                initcoeff = initcoeff_linear
                print initcoeff

            #generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, initcoeff_linear),(zoomfactory,zoomfactorx))
            #generated_intensity /= generated_intensity.max()
            #plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
            #plt.show() 
            #initcoeff = np.array([0,0,0,0,0,0])
            iternumber = 0
            while 1:
                print 'iternumber =', iternumber,'for',yy,xx
                result = basinhopping(nl, initcoeff, niter = 5, T=100, stepsize=2e-5, interval=50,accept_test=accept_test,minimizer_kwargs={'method': 'Nelder-Mead', 'args': (data_patch,(zoomfactory,zoomfactorx))}, disp=True, callback=callback)
                print result.fun
                if result.fun <threshold:
                    xopt = result.x
                    break
                else:
                    initcoeff = result.x
                    iternumber+=1
                    if iternumber == 20:
                        xopt = initcoeff_extend 
                        break
                        initcoeff_extend = initcoeff_linear
                        #print 'using linear coefficients'
                    #if iternumber == 20:
                    #    xopt = initcoeff_extend
                    #    break

            #print xopt

            generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, xopt,(zoomfactory,zoomfactorx)))
            generated_intensity /= generated_intensity.max()
            #plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
            #plt.show()
            generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
            fitimg[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity

            clist = []
            if (int(yy/dyy),int(xx/dxx)-1) in hstore_upperright:
                print 'found upperright'
                clist.append(hstore_upperright[(int(yy/dyy),int(xx/dxx)-1)])
            if (int(yy/dyy)-1,int(xx/dxx)) in hstore_lowerleft:
                print 'found lowerleft'
                clist.append(hstore_lowerleft[(int(yy/dyy)-1,int(xx/dxx))])
            if (int(yy/dyy)-1,int(xx/dxx)-1) in hstore_lowerright:
                print 'found lowerright'
                clist.append(hstore_lowerright[(int(yy/dyy)-1,int(xx/dxx)-1)])
            if len(clist)>0:
                print 'clist=', clist
                if max(clist)-np.median(clist)>0.532/2:
                    clist.remove(max(clist))
                    print 'maxremove'
                if np.median(clist)-min(clist)>0.532/2:
                    clist.remove(min(clist))
                    print 'minremove'
                xopt[5] = np.mean(clist)

            height = surface_polynomial(data_patch.shape, xopt,(zoomfactory,zoomfactorx))
            hupperright = height[0,-1]
            hlowerright = height[-1,-1]
            hlowerleft = height[-1,0]
            if iternumber <20:
                print 'coeff & corner heights stored'
                xstore[(int(yy/dyy),int(xx/dxx))]=xopt
                hstore_upperright[(int(yy/dyy),int(xx/dxx))] = hupperright
                hstore_lowerright[(int(yy/dyy),int(xx/dxx))] = hlowerright
                hstore_lowerleft[(int(yy/dyy),int(xx/dxx))] = hlowerleft
            else:
                xstore_badtiles[(int(yy/dyy),int(xx/dxx))]=xopt
                print (int(yy/dyy),int(xx/dxx)), 'is a bad tile'
            X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
            ax.plot_wireframe(X,Y,height,rstride=20,cstride=20)
            ax.set_aspect('equal')
            plt.draw()
            plt.pause(0.01)
            cv2.imwrite('fitimg.tif', fitimg.astype('uint8'))
            print '\n'
    np.save('xoptstore'+strftime("%Y%m%d_%H_%M_%S",localtime()),xstore)
    np.save('xoptstore_badtiles'+strftime("%Y%m%d_%H_%M_%S",localtime()),xstore_badtiles)
    print 'time used', time.time()-start, 's'
    print 'finished'
    plt.show()
