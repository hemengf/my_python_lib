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
from scipy.ndimage import gaussian_filter

def equalize(img_array):
    """
    returns array with float 0-1

    """
    img_array = img_array/(img_array.max()+1e-6)
    #equalized = exposure.equalize_adapthist(img_array,kernal_size = (5,5))
    equalized = exposure.equalize_hist(img_array)
    #equalized = img_array/img_array.max()
    return equalized 

def difference(data_img, generated_img,mask_patch):
    """
    both images have to be 0-1float

    """
    data_img = gaussian_filter(data_img,sigma=0.3)
    generated_img = gaussian_filter(generated_img, sigma=0)
    diff_value = np.sum(mask_patch*(data_img-generated_img)**2)
    diff_value /= (mask_patch.sum())#percentage of white area
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

def nl(coeff, data_img,(zoomfactory,zoomfactorx),mask_patch):
    """
    negative likelyhood-like function; aim to minimize this
    data_img has to be 0-1float
    
    """
    height = surface_polynomial(data_img.shape,coeff,(zoomfactory,zoomfactorx))
    expected= 1+ np.cos((4*np.pi/0.532)*height)
    expected /= expected.max()#normalize to 0-1float
    #expected = equalize(expected)
    return difference(data_img, expected,mask_patch)

def accept_test(f_new,x_new,f_old,x_old):
    return True
    if abs(x_new[3])>0.05 or abs(x_new[4])>0.05:
        return False
    else:
        return True

def callback(x,f,accept):
    #print x[3],x[4],f,accept
    pass

def find_tilequeue4(processed_tiles):
    tilequeue = []
    for tile in processed_tiles:
        tilequeue.append((tile[0]+1,tile[1])) #right
        tilequeue.append((tile[0]-1,tile[1])) #left
        tilequeue.append((tile[0],tile[1]+1)) #down
        tilequeue.append((tile[0],tile[1]-1)) #up
        #tilequeue.append((tile[0]+1,tile[1]-1)) #upperright
        #tilequeue.append((tile[0]-1,tile[1]+1)) #lowerleft
        #tilequeue.append((tile[0]+1,tile[1]+1)) #lowerright
        #tilequeue.append((tile[0]-1,tile[1]-1)) #upperleft
    tilequeue = [tile for tile in tilequeue if tile not in processed_tiles]
    return list(set(tilequeue))

def find_tilequeue8(processed_tiles):
    tilequeue = []
    for tile in processed_tiles:
        tilequeue.append((tile[0]+1,tile[1])) #right
        tilequeue.append((tile[0]-1,tile[1])) #left
        tilequeue.append((tile[0],tile[1]+1)) #down
        tilequeue.append((tile[0],tile[1]-1)) #up
        tilequeue.append((tile[0]+1,tile[1]-1)) #upperright
        tilequeue.append((tile[0]-1,tile[1]+1)) #lowerleft
        tilequeue.append((tile[0]+1,tile[1]+1)) #lowerright
        tilequeue.append((tile[0]-1,tile[1]-1)) #upperleft
    tilequeue = [tile for tile in tilequeue if tile not in processed_tiles]
    return list(set(tilequeue))

def fittile(tile, dxx,dyy,zoomfactorx, zoomfactory, data_img, mask_img,xstore, abquadrant, white_threshold):
    yy = tile[0]*dyy 
    xx = tile[1]*dxx 
    data_patch = data_img[yy:yy+dyy,xx:xx+dxx]
    data_patch = data_patch[::zoomfactory,::zoomfactorx]

    mask_patch = mask_img[yy:yy+dyy,xx:xx+dxx]
    mask_patch = mask_patch[::zoomfactory,::zoomfactorx]

    data_patch= equalize(data_patch)#float0-1
    white_percentage = (mask_patch.sum()/len(mask_patch.flat))
    if white_percentage < white_threshold:
        goodness = threshold/white_percentage
        return [np.nan,np.nan,np.nan,np.nan,np.nan, np.nan],goodness, white_percentage
    initcoeff_extendlist = []

    if (int(yy/dyy)-1,int(xx/dxx)) in xstore:
        #print 'found up'
        up = xstore[(int(yy/dyy)-1,int(xx/dxx))]
        initcoeff_extendlist.append(np.array([up[0],up[1],up[2],up[2]*dyy+up[3],2*up[1]*dyy+up[4],up[1]*dyy*dyy+up[4]*dyy+up[5]]))
    if (int(yy/dyy)+1,int(xx/dxx)) in xstore:
        #print 'found down'
        up = xstore[(int(yy/dyy)+1,int(xx/dxx))]
        initcoeff_extendlist.append(np.array([up[0],up[1],up[2],-up[2]*dyy+up[3],-2*up[1]*dyy+up[4],up[1]*dyy*dyy-up[4]*dyy+up[5]]))
    if (int(yy/dyy),int(xx/dxx)-1) in xstore:
        #print 'found left'
        left = xstore[(int(yy/dyy),int(xx/dxx)-1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx+left[3],left[2]*dxx+left[4],left[0]*dxx*dxx+left[3]*dxx+left[5]]))
    if (int(yy/dyy),int(xx/dxx)+1) in xstore:
        #print 'found right'
        left = xstore[(int(yy/dyy),int(xx/dxx)+1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx+left[3],-left[2]*dxx+left[4],left[0]*dxx*dxx-left[3]*dxx+left[5]]))

    if (int(yy/dyy)-1,int(xx/dxx)-1) in xstore:
        #print 'found upperleft'
        left = xstore[(int(yy/dyy)-1,int(xx/dxx)-1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx+left[2]*dyy+left[3],left[2]*dxx+2*left[1]*dyy+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy+left[2]*dxx*dyy+left[3]*dxx+left[4]*dyy+left[5]]))
    if (int(yy/dyy)+1,int(xx/dxx)-1) in xstore:
        #print 'found lowerleft'
        left = xstore[(int(yy/dyy)+1,int(xx/dxx)-1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx-left[2]*dyy+left[3],left[2]*dxx-2*left[1]*dyy+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy-left[2]*dxx*dyy+left[3]*dxx-left[4]*dyy+left[5]]))
    if (int(yy/dyy)+1,int(xx/dxx)+1) in xstore:
        #print 'found lowerright'
        left = xstore[(int(yy/dyy)+1,int(xx/dxx)+1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx-left[2]*dyy+left[3],-left[2]*dxx-2*left[1]*dyy+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy+left[2]*dxx*dyy-left[3]*dxx-left[4]*dyy+left[5]]))
    if (int(yy/dyy)-1,int(xx/dxx)+1) in xstore:
        #print 'found upperright'
        left = xstore[(int(yy/dyy)-1,int(xx/dxx)+1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx+left[2]*dyy+left[3],-left[2]*dxx+2*left[1]*dyy+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy-left[2]*dxx*dyy-left[3]*dxx+left[4]*dyy+left[5]]))

        """
#######################################################
    if (int(yy/dyy)-2,int(xx/dxx)) in xstore:
        #print 'found up'
        up = xstore[(int(yy/dyy)-2,int(xx/dxx))]
        initcoeff_extendlist.append(np.array([up[0],up[1],up[2],up[2]*dyy*2+up[3],2*up[1]*dyy*2+up[4],up[1]*dyy*dyy*4+up[4]*dyy*2+up[5]]))
    if (int(yy/dyy)+2,int(xx/dxx)) in xstore:
        #print 'found down'
        up = xstore[(int(yy/dyy)+2,int(xx/dxx))]
        initcoeff_extendlist.append(np.array([up[0],up[1],up[2],-up[2]*dyy*2+up[3],-2*up[1]*dyy*2+up[4],up[1]*dyy*dyy*4-up[4]*dyy*2+up[5]]))
    if (int(yy/dyy),int(xx/dxx)-2) in xstore:
        #print 'found left'
        left = xstore[(int(yy/dyy),int(xx/dxx)-2)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx*2+left[3],left[2]*dxx*2+left[4],left[0]*dxx*dxx*4+left[3]*dxx*2+left[5]]))
    if (int(yy/dyy),int(xx/dxx)+2) in xstore:
        #print 'found right'
        left = xstore[(int(yy/dyy),int(xx/dxx)+2)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx*2+left[3],-left[2]*dxx*2+left[4],left[0]*dxx*dxx*4-left[3]*dxx*2+left[5]]))
    if (int(yy/dyy)-2,int(xx/dxx)-1) in xstore:
        #print 'found upperleft'
        left = xstore[(int(yy/dyy)-2,int(xx/dxx)-1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx+left[2]*dyy*2+left[3],left[2]*dxx+2*left[1]*dyy*2+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy*4+left[2]*dxx*dyy*2+left[3]*dxx+left[4]*dyy*2+left[5]]))
    if (int(yy/dyy)-1,int(xx/dxx)-2) in xstore:
        #print 'found upperleft'
        left = xstore[(int(yy/dyy)-1,int(xx/dxx)-2)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx*2+left[2]*dyy+left[3],left[2]*dxx*2+2*left[1]*dyy+left[4],left[0]*dxx*dxx*4+left[1]*dyy*dyy+left[2]*dxx*2*dyy+left[3]*dxx*2+left[4]*dyy+left[5]]))
    if (int(yy/dyy)+2,int(xx/dxx)-1) in xstore:
        #print 'found lowerleft'
        left = xstore[(int(yy/dyy)+2,int(xx/dxx)-1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx-left[2]*dyy*2+left[3],left[2]*dxx-2*left[1]*dyy*2+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy*4-left[2]*dxx*dyy*2+left[3]*dxx-left[4]*dyy*2+left[5]]))
    if (int(yy/dyy)+1,int(xx/dxx)-2) in xstore:
        #print 'found lowerleft'
        left = xstore[(int(yy/dyy)+1,int(xx/dxx)-2)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],2*left[0]*dxx*2-left[2]*dyy+left[3],left[2]*dxx*2-2*left[1]*dyy+left[4],left[0]*dxx*dxx*4+left[1]*dyy*dyy-left[2]*dxx*2*dyy+left[3]*dxx*2-left[4]*dyy+left[5]]))
    if (int(yy/dyy)+1,int(xx/dxx)+2) in xstore:
        #print 'found lowerright'
        left = xstore[(int(yy/dyy)+1,int(xx/dxx)+2)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx*2-left[2]*dyy+left[3],-left[2]*dxx*2-2*left[1]*dyy+left[4],left[0]*dxx*dxx*2+left[1]*dyy*dyy+left[2]*dxx*2*dyy-left[3]*dxx*2-left[4]*dyy+left[5]]))
    if (int(yy/dyy)+2,int(xx/dxx)+1) in xstore:
        #print 'found lowerright'
        left = xstore[(int(yy/dyy)+2,int(xx/dxx)+1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx-left[2]*dyy*2+left[3],-left[2]*dxx-2*left[1]*dyy*2+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy*2+left[2]*dxx*dyy*2-left[3]*dxx-left[4]*dyy*2+left[5]]))
    if (int(yy/dyy)-2,int(xx/dxx)+1) in xstore:
        #print 'found upperright'
        left = xstore[(int(yy/dyy)-2,int(xx/dxx)+1)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx+left[2]*dyy*2+left[3],-left[2]*dxx+2*left[1]*dyy*2+left[4],left[0]*dxx*dxx+left[1]*dyy*dyy*4-left[2]*dxx*dyy*2-left[3]*dxx+left[4]*dyy*2+left[5]]))
    if (int(yy/dyy)-1,int(xx/dxx)+2) in xstore:
        #print 'found upperright'
        left = xstore[(int(yy/dyy)-1,int(xx/dxx)+2)]
        initcoeff_extendlist.append(np.array([left[0],left[1],left[2],-2*left[0]*dxx*2+left[2]*dyy+left[3],-left[2]*dxx*2+2*left[1]*dyy+left[4],left[0]*dxx*dxx*2+left[1]*dyy*dyy-left[2]*dxx*2*dyy-left[3]*dxx*2+left[4]*dyy+left[5]]))
###############################################################
    """

    if len(initcoeff_extendlist) > 0:
        initcoeff_extend = np.mean(initcoeff_extendlist,axis=0)
        initcoeff = initcoeff_extend
    else: #if no touching tiles are detected, should be only for the starting tile
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
                    nl_1storder[j,i] = nl([0,0,0,aa[j,i],bb[j,i],0],data_patch,(zoomfactory,zoomfactorx),mask_patch)
                sys.stdout.write('\r%i/%i     ' % (i*blist.size+j+1,alist.size*blist.size))
                sys.stdout.flush()
        sys.stdout.write('\n')
        index = np.unravel_index(np.nanargmin(nl_1storder), nl_1storder.shape)
        index = (alist[index[1]], blist[index[0]])
        index = np.array(index)
        initcoeff_linear= np.array([0,0,0,index[0],index[1],0])
        initcoeff = initcoeff_linear
        print initcoeff
    iternumber = 0
    while 1:
        #print 'iternumber =', iternumber,'for',yy,xx
        result = basinhopping(nl, initcoeff, niter = 8, T=0.01, stepsize=5e-5, interval=50,accept_test=accept_test,minimizer_kwargs={'method': 'Nelder-Mead', 'args': (data_patch,(zoomfactory,zoomfactorx), mask_patch)}, disp=False, callback=callback)
        print result.fun
        if result.fun <threshold:
            xopt = result.x
            break
        else:
            initcoeff = result.x
            iternumber+=1
            if iternumber == 5:
                xopt = initcoeff_extend 
                break
    goodness = result.fun
    return xopt, goodness, white_percentage


def tilewithinbound(tile, dxx, dyy, data_img):
    if tile[0]<0 or tile[1]<0:
        return False
    elif (tile[1]+1)*dxx>data_img.shape[1] or (tile[0]+1)*dyy>data_img.shape[0]:
        return False
    else:
        return True

if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter
    import time
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from time import localtime, strftime

    start = time.time()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    N = 100 #a,b value resolution; a, b linear term coeff
    sample_size = 0.2 #a, b value range
    abquadrant = 3
    data_img = cv2.imread('sample4.tif', 0)
    mask_img = cv2.imread('mask_bot_v2.tif', 0)

    data_img = data_img.astype('float64')
    mask_img = mask_img.astype('float64')
    mask_img /= 255.
    fitimg = np.copy(data_img)
    xstore = {}
    xstore_badtiles = {}
    hstore_upperright = {}
    hstore_lowerright = {}
    hstore_lowerleft = {}
    hstore_upperleft= {}
    dyy,dxx = 81,81
    threshold = 0.08
    white_threshold = 0.4
    startingposition = (928,2192)
    startingtile =  (int(startingposition[0]/dyy),int(startingposition[1]/dxx))
    zoomfactory,zoomfactorx = 1,1
    tilequeue = find_tilequeue8([startingtile])
    tilequeue = [startingtile]+tilequeue
    processed_tiles = []
    bad_tiles= []
    black_tiles= []
    tilequeue = [tile for tile in tilequeue if tilewithinbound(tile,dxx, dyy, data_img)]
    goodness_dict= {}
    while any(tilequeue): 
        print tilequeue
        # check queue for a collection of goodness and get the best tile
        for tile in tilequeue:
            if tile not in goodness_dict: #avoid double checking the tiles shared by the old tilequeue
                print 'prechecking tile: ',tile
                xopttrial, goodness,white_percentage = fittile(tile,dxx,dyy,zoomfactorx, zoomfactory, data_img, mask_img,xstore,abquadrant,white_threshold)
                print 'white percentage:', white_percentage
                goodness_dict[tile] = goodness
                if white_percentage >= white_threshold:
                    if goodness <= threshold:
                        xstore[tile] = xopttrial
                    elif goodness > threshold:
                        bad_tiles.append(tile) #never used it
                        print 'bad tile:', tile
                else:
                    black_tiles.append(tile)
                    print 'black tile:', tile
        goodness_queue = {tile:goodness_dict[tile] for tile in tilequeue}
        best_tile = min(goodness_queue,key=goodness_queue.get) 

        yy,xx  = best_tile[0]*dyy, best_tile[1]*dxx 

        print 'processing best tile', (int(yy/dyy),int(xx/dxx)) 


        processed_tiles.append((int(yy/dyy),int(xx/dxx)))#update processed tiles
        tilequeue = find_tilequeue8(processed_tiles)#update tilequeue
        tilequeue = [tile for tile in tilequeue if tilewithinbound(tile,dxx, dyy, data_img)]

        if best_tile in black_tiles:
            break


        data_patch = data_img[yy:yy+dyy,xx:xx+dxx]
        data_patch = data_patch[::zoomfactory,::zoomfactorx]

        mask_patch = mask_img[yy:yy+dyy,xx:xx+dxx]
        mask_patch = mask_patch[::zoomfactory,::zoomfactorx]

        data_patch= equalize(data_patch)#float0-1

        xopt, goodness, white_percentage = fittile(best_tile, dxx,dyy,zoomfactorx, zoomfactory, data_img, mask_img,xstore, abquadrant, white_threshold)

        generated_intensity = 1+np.cos((4*np.pi/0.532)*surface_polynomial(data_patch.shape, xopt,(zoomfactory,zoomfactorx)))
        generated_intensity /= generated_intensity.max()
        #plt.imshow(np.concatenate((generated_intensity,data_patch,(generated_intensity-data_patch)**2),axis=1))
        #plt.show()
        generated_intensity = zoom(generated_intensity,(zoomfactory,zoomfactorx))
        fitimg[yy:yy+dyy,xx:xx+dxx] = 255*generated_intensity
        if best_tile in bad_tiles:
            fitimg[yy:yy+5,xx:xx+dxx] = 0
            fitimg[yy+dyy-5:yy+dyy,xx:xx+dxx] = 0
            fitimg[yy:yy+dyy,xx:xx+5] = 0
            fitimg[yy:yy+dyy,xx+dxx-5:xx+dxx] = 0

        height = surface_polynomial(data_patch.shape, xopt,(zoomfactory,zoomfactorx))
        hupperright = height[0,-1]
        hlowerright = height[-1,-1]
        hlowerleft = height[-1,0]
        hupperleft = height[0,0]

        clist = []
        #upperleft node
        if (int(yy/dyy),int(xx/dxx)-1) in hstore_upperright:
            clist.append(hstore_upperright[(int(yy/dyy),int(xx/dxx)-1)])
        if (int(yy/dyy)-1,int(xx/dxx)) in hstore_lowerleft:
            clist.append(hstore_lowerleft[(int(yy/dyy)-1,int(xx/dxx))])
        if (int(yy/dyy)-1,int(xx/dxx)-1) in hstore_lowerright:
            clist.append(hstore_lowerright[(int(yy/dyy)-1,int(xx/dxx)-1)])
        #lowerleft node
        if (int(yy/dyy),int(xx/dxx)-1) in hstore_lowerright:
            correction_to_currentc = hstore_lowerright[(int(yy/dyy),int(xx/dxx)-1)]-hlowerleft
            clist.append(xopt[5]+correction_to_currentc)
        if (int(yy/dyy)+1,int(xx/dxx)-1) in hstore_upperright:
            correction_to_currentc = hstore_upperright[(int(yy/dyy)+1,int(xx/dxx)-1)]-hlowerleft
            clist.append(xopt[5]+correction_to_currentc)
        if (int(yy/dyy)+1,int(xx/dxx)) in hstore_upperleft:
            correction_to_currentc = hstore_upperleft[(int(yy/dyy)+1,int(xx/dxx))]-hlowerleft
            clist.append(xopt[5]+correction_to_currentc)
        #lowerright node
        if (int(yy/dyy),int(xx/dxx)+1) in hstore_lowerleft:
            correction_to_currentc = hstore_lowerleft[(int(yy/dyy),int(xx/dxx)+1)]-hlowerright
            clist.append(xopt[5]+correction_to_currentc)
        if (int(yy/dyy)+1,int(xx/dxx)+1) in hstore_upperleft:
            correction_to_currentc = hstore_upperleft[(int(yy/dyy)+1,int(xx/dxx)+1)]-hlowerright
            clist.append(xopt[5]+correction_to_currentc)
        if (int(yy/dyy)+1,int(xx/dxx)) in hstore_upperright:
            correction_to_currentc = hstore_upperright[(int(yy/dyy)+1,int(xx/dxx))]-hlowerright
            clist.append(xopt[5]+correction_to_currentc)
        #upperright node
        if (int(yy/dyy),int(xx/dxx)+1) in hstore_upperleft:
            correction_to_currentc = hstore_upperleft[(int(yy/dyy),int(xx/dxx)+1)]-hupperright
            clist.append(xopt[5]+correction_to_currentc)
        if (int(yy/dyy)-1,int(xx/dxx)+1) in hstore_lowerleft:
            correction_to_currentc = hstore_lowerleft[(int(yy/dyy)-1,int(xx/dxx)+1)]-hupperright
            clist.append(xopt[5]+correction_to_currentc)
        if (int(yy/dyy)-1,int(xx/dxx)) in hstore_lowerright:
            correction_to_currentc = hstore_lowerright[(int(yy/dyy)-1,int(xx/dxx))]-hupperright
            clist.append(xopt[5]+correction_to_currentc)
            
        if len(clist)>0:
            #print 'clist=', clist
            #if max(clist)-np.median(clist)>0.532/2:
            #    clist.remove(max(clist))
            #    print 'maxremove'
            #if np.median(clist)-min(clist)>0.532/2:
            #    clist.remove(min(clist))
            #    print 'minremove'
            xopt[5] = np.mean(clist)

        height = surface_polynomial(data_patch.shape, xopt,(zoomfactory,zoomfactorx))
        hupperright = height[0,-1]
        hlowerright = height[-1,-1]
        hlowerleft = height[-1,0]
        hupperleft = height[0,0]

        #if iternumber <20:
        if 1:
            #print 'coeff & corner heights stored'
            xstore[(int(yy/dyy),int(xx/dxx))]=xopt
            hstore_upperright[(int(yy/dyy),int(xx/dxx))] = hupperright
            hstore_lowerright[(int(yy/dyy),int(xx/dxx))] = hlowerright
            hstore_lowerleft[(int(yy/dyy),int(xx/dxx))] = hlowerleft
            hstore_upperleft[(int(yy/dyy),int(xx/dxx))] = hupperleft
        else:
            xstore_badtiles[(int(yy/dyy),int(xx/dxx))]=xopt
            print (int(yy/dyy),int(xx/dxx)), 'is a bad tile'
        X,Y =np.meshgrid(range(xx,xx+dxx,zoomfactorx),range(data_img.shape[0]-yy,data_img.shape[0]-yy-dyy,-zoomfactory))
        ax.plot_wireframe(X,Y,height,rstride=int(dyy/1),cstride=int(dxx/1))
        ax.set_aspect('equal')
        plt.draw()
        plt.pause(0.01)
        cv2.imwrite('fitimg_bot.tif', fitimg.astype('uint8'))
        print '\n'
    np.save('xoptstore_bot',xstore)
    #np.save('xoptstore_badtiles'+strftime("%Y%m%d_%H_%M_%S",localtime()),xstore_badtiles)
    print 'time used', time.time()-start, 's'
    print 'finished'
    plt.show()
