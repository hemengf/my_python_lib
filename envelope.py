from __future__ import division
import numpy as np
from scipy.signal import savgol_filter as sg
from scipy.interpolate import interp1d
from skimage.measure import profile_line as pl
from find_peaks import left_find_indices_min as minindices
from find_peaks import left_find_indices_max as maxindices
import sys
import time
import os

def meandata(img,(startx,starty)=(2042,1674),R=1067,a=167,da=20,dda=1,savename="mdatatemp"):
    """ 
    R profile length 
    a angle
    da averaging angle
    dda averaging stepping size
    """
    if os.path.exists(savename):
        data = np.load(savename)
        mdata = np.mean(data,axis=0)
    else:
        for i,angle in enumerate(np.arange(a,a+da,dda)):
            endx = startx+np.cos(angle*np.pi/180)*R
            endy = starty-np.sin(angle*np.pi/180)*R
            #endx,endy pixel/imagej coord, need to reverse for scipy/numpy use
            if i == 0:
                data =  pl(img,(starty,startx),(endy,endx),order=0)
                length = len(data)
            else:
                start = time.time()
                data = np.vstack((data,pl(img,(starty,startx),(endy,endx),order = 3)[:length]))
                sys.stdout.write("averaging: %d/%d, takes %fs\r"%(i+1,len(np.arange(a,a+da,dda)),time.time()-start))
        np.save(savename,data)
        mdata = np.mean(data,axis=0)
    return mdata
def normalize_envelope(mdata,smoothwindow=19,splineorder=2,envelopeinterp='quadratic'):
    """
    x is the maximum range where envelop fitting is possible
    """
    s = sg(mdata,smoothwindow,splineorder)
    upperx = maxindices(s)
    #uppery = np.maximum(mdata[upperx],s[upperx])
    uppery = mdata[upperx]
    lowerx = minindices(s)
    #lowery = np.minimum(mdata[lowerx],s[lowerx])
    lowery = mdata[lowerx]
    fupper = interp1d(upperx, uppery, kind=envelopeinterp)
    flower = interp1d(lowerx, lowery, kind=envelopeinterp)
    x = np.arange(max(min(upperx),min(lowerx)),min(max(upperx),max(lowerx)))
    y = mdata[x]
    newy = (y-flower(x))/(fupper(x)-flower(x))
    return x,newy

if __name__=="__main__":
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    (startx,starty)=(2042,1674)
    R = 1067
    a = 167
    da = 20
    dda = 1


    imgred = cv2.imread('warpedred.tif',0)
    imggreen = cv2.imread('warpedgreen.tif',0)
    imgamber = cv2.imread('DSC_3878.jpg',0) 
    cmap = plt.get_cmap('tab10')
    am = cmap(1)
    gr = cmap(2)
    rd = cmap(3)

    print '\nprocessing red'
    mdatared = meandata(imgred,(startx,starty),R,a,da,dda,savename='datared.npy')
    xred,newyred = normalize_envelope(mdatared[170:]) #170 is to cut off the flat noisy first dark spot; otherwise envelope fitting won't work (it assumes a nice wavy shape without too many local extrema)
    xred+=170 #not necessary; just to make sure xred=0 is center of the rings;wanna make sure all the coordinates throughout the script is consistent so its easier to check for bugs

    print '\nprocessing amber'
    mdataamber = meandata(imgamber,(startx,starty),R,a,da,dda,savename='dataamber.npy')
    xamber,newyamber = normalize_envelope(mdataamber[170:])
    xamber+=170

    print'\nprocess green'
    mdatagreen= meandata(imggreen, (startx,starty),R,a,da,dda,savename='datagreen.npy')
    xgreen,newygreen= normalize_envelope(mdatagreen[170:])
    xgreen+=170
    np.save('xgreen',xgreen)
    np.save('newygreen',newygreen)

    #plt.plot(mdatared,color=cmap(3))
    #plt.plot(mdatagreen,color=cmap(2))
    #plt.plot(mdataamber,color=cmap(1))
    plt.plot(xred,newyred,color=rd)
    plt.plot(xamber,newyamber,color=am)
    plt.plot(xgreen,newygreen,color=gr)
    plt.show()
