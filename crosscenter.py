from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
from collections import namedtuple

def roughcenter(img,ilwindow,jlwindow,i0,j0):
    """ Returns icenter, jcenter only using 4 tips of the cross shape.
    
    img needs to be blurred;
    Starts from i0, j0, draws a window of height and width of lwindow, jlwindow; 
    Gets 4 intersections with the window edge; 
    Gets ic, jc by cross connecting the 4 intersection points.
    """
    edge1 = img[i0-int(ilwindow/2) : i0+int(ilwindow/2), j0-int(jlwindow/2)]
    indx = np.argmin(edge1)
    i1, j1 = i0-int(ilwindow/2)+indx, j0-int(jlwindow/2)
    x1, y1 = j1,i1

    edge2 = img[i0-int(ilwindow/2) , j0-int(jlwindow/2) : j0+int(jlwindow/2)]
    indx = np.argmin(edge2)
    i2, j2 = i0-int(ilwindow/2), j0-int(jlwindow/2)+indx
    x2, y2 = j2,i2

    edge3 = img[i0-int(ilwindow/2) : i0+int(ilwindow/2) , j0+int(jlwindow/2)]
    indx = np.argmin(edge3)
    i3, j3 = i0-int(ilwindow/2)+indx, j0+int(jlwindow/2)
    x3, y3 = j3,i3

    edge4 = img[i0+int(ilwindow/2) ,j0-int(jlwindow/2) : j0+int(jlwindow/2)]
    indx = np.argmin(edge4)
    i4, j4 = i0+int(ilwindow/2), j0-int(jlwindow/2)+indx
    x4, y4 = j4,i4
    
    if (x2 == x4) or (y1 == y3):
        xc = x2 
        yc = y1
    else:
        s13 = (y3-y1)/(x3-x1)
        s24 = (y4-y2)/(x4-x2)
        yc = (s13*s24*(x2-x1) + s24*y1-s13*y2)/(s24-s13)
        xc = (yc-y1)/s13+x1

    ic,jc = int(yc),int(xc)
    Res = namedtuple('Res','xc,yc,ic,jc,i1,j1,i2,j2,i3,j3,i4,j4')
    res = Res(xc, yc, ic, jc, i1,j1, i2, j2, i3, j3, i4, j4)
    return res 

def mixture_lin(img,ilwindow,jlwindow,i0,j0,thresh):
    """Returns xcenter, ycenter of a cross shape using mixture linear regression.

    img doesn't have to be bw; but training points are 0 intensity;
    ilwindow, jlwindow,i0,j0 for target area;
    Use thresh (e.g., 0.6) to threshold classification;
    Best for two bars making a nearly vertical crossing.
    """
    img = img[i0-int(ilwindow/2):i0+int(ilwindow/2), j0-int(jlwindow/2):j0+int(jlwindow/2)]
    X_train = np.argwhere(img == 0 )
    n = np.shape(X_train)[0]  #number of points
    y = X_train[:,0] 
    x = X_train[:,1]

    w1 = np.random.normal(0.5,0.1,n)
    w2 = 1-w1

    start = time.time()
    for i in range(100):
        pi1_new = np.mean(w1) 
        pi2_new = np.mean(w2) 

        mod1= sm.WLS(y,sm.add_constant(x),weights = w1) #vertical
        res1 = mod1.fit()

        mod2= sm.WLS(x,sm.add_constant(y),weights = w2) #horizontal
        res2 = mod2.fit()

        y1_pred_new= res1.predict(sm.add_constant(x)) 
        sigmasq1 = np.sum(res1.resid**2)/n
        a1 = pi1_new * np.exp((-(y-y1_pred_new)**2)/sigmasq1)

        x2_pred_new = res2.predict(sm.add_constant(y)) 
        sigmasq2 = np.sum(res2.resid**2)/n
        a2 = pi2_new * np.exp((-(x-x2_pred_new)**2)/sigmasq2)

        if np.max(abs(a1/(a1+a2)-w1))<1e-5:
            #print '%d iterations'%i
            break

        w1 = a1/(a1+a2)
        w2 = a2/(a1+a2)
    #print '%.3fs'%(time.time()-start)
    #plt.scatter(x, y,10, c=w1,cmap='RdBu')
    #w1thresh = (w1>thresh)+0
    #w2thresh = (w2>thresh)+0

    x1 = x[w1>thresh]
    x2 = x[w2>thresh]
    y1 = y[w1>thresh]
    y2 = y[w2>thresh]

    mod1 = sm.OLS(y1,sm.add_constant(x1))
    res1 = mod1.fit()
    sigmasq1 = np.sum(res1.resid**2)/len(x1)
    y1_pred= res1.predict(sm.add_constant(x1)) 
    #plt.plot(x1, y1_pred)

    mod2 = sm.OLS(x2,sm.add_constant(y2))
    res2 = mod2.fit()
    sigmasq2= np.sum(res2.resid**2)/len(x2)
    x2_pred= res2.predict(sm.add_constant(y2)) 
    #plt.plot(x2_pred,y2)

    b1,k1 = res1.params # y = k1x + b1
    b2,k2 = res2.params # x = k2y + b2
    yc = (k1*b2+b1)/(1-k1*k2)
    xc = k2*yc + b2
    #plt.scatter(xc,yc)
    # all above values are wrt small cropped picture
    xc += j0-jlwindow/2
    x1 = x1 + j0-jlwindow/2
    x2_pred = x2_pred + j0-jlwindow/2
    yc += i0-ilwindow/2
    y1_pred = y1_pred + i0-ilwindow/2
    y2 = y2 + i0-ilwindow/2

    Res = namedtuple('Res','xc, yc,x1,y1_pred,x2_pred,y2,sigmasq1,sigmasq2')
    res = Res(xc, yc,x1,y1_pred,x2_pred,y2,sigmasq1,sigmasq2)
    return res

if __name__ == "__main__":
    img = cv2.imread('c:/Users/Mengfei/nagellab/forcedwetting/velocity_tracking/sample8.tif',0)
    (_, img) = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    thresh = 0.6
    ilwindow,jlwindow = 50, 50 
    x0, y0 = 421,371 
    i0, j0 = y0,x0 

    res = mixture_lin(img,ilwindow,jlwindow, i0,j0,thresh)
    print res.sigmasq1

    plt.imshow(img,'gray')
    plt.scatter(res.xc,res.yc)
    plt.plot(res.x1,res.y1_pred)
    plt.plot(res.x2_pred,res.y2)
    plt.show()
