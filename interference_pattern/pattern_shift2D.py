from __future__ import division
import scipy.optimize
import scipy.spatial.distance
import partial_derivative
import math

@profile
def shape_function(x,y):
    #return np.exp(-0.00002*((x+250)**2+y**2)) + np.exp(-0.00002*((x-250)**2+y**2))+100
    return 0.000005*(x**2+y**2)+68
    #return 0.00000001*x + 68

@profile
def find_k_refracting(k_incident, x1, n1,n2):
    # x1 in the form [x1,y1]
    gradient = partial_derivative.partial_derivative(shape_function, *x1)
    # gradient in the form [df/dx1,df/dy1]
    #n = np.r_[-gradient, 1] adding a column in memory is too slow
    n = np.empty((3,))
    n[:-1] = -gradient
    n[-1] = 1
    #print "n = ", n
    #print "x1 = ", x1
    norm =np.linalg.norm(n)
    n = n/norm  # n is the unit normal vector pointing 'upward'
    c = -np.dot(n, k_incident)
    r = n1/n2
    sqrtterm = (1-r**2*(1-c**2)) 
    if sqrtterm < 0:
        print(Fore.RED)
        print "Total internal reflection occurred."
        print "1-r**2*(1-c**2) = \n", sqrtterm 
        print(Style.RESET_ALL)
        sys.exit(0)
    factor = (r*c- math.sqrt(sqrtterm))
    k_refracting = r*k_incident + factor*n
    #print 'c =',c 
    #print "factor", factor 
    #print "k_refracting = ", k_refracting
    return k_refracting

@profile
def find_x0(k_incident, x1, n1,n2):
#    def Fx(x):
#        k_refracting = find_k_refracting(k_incident, x, n1, n2)
#        return k_refracting[0]*(shape_function(*x1)+shape_function(*x))+k_refracting[2]*(x1-x)[0]
#    def Fy(x):
#        k_refracting = find_k_refracting(k_incident, x, n1, n2)
#        return k_refracting[1]*(shape_function(*x1)+shape_function(*x))+k_refracting[2]*(x1-x)[1]
#    def F(x):
#        return Fx(x), Fy(x)
    def F(x):
        k_refracting = find_k_refracting(k_incident, x, n1, n2)
        return k_refracting[0]*(shape_function(*x1)+shape_function(*x))+k_refracting[2]*(x1-x)[0], k_refracting[1]*(shape_function(*x1)+shape_function(*x))+k_refracting[2]*(x1-x)[1]
    sol = scipy.optimize.root(F,x1) 
    x0 = sol.x
    return x0

@profile
def optical_path_diff(k_incident, x1, n1,n2):
    x0 = find_x0(k_incident, x1, n1, n2)
    p0 = np.empty((3,))
    p1 = np.empty((3,))
    p1_image_point = np.empty((3,))
    p0[:-1] = x0
    p1[:-1] = x1
    p1_image_point[:-1] = x1
    p0[-1] = shape_function(*x0)
    p1[-1] = shape_function(*x1)
    p1_image_point[-1] = -shape_function(*x1)
    #p0 = np.r_[x0, shape_function(*x0)]
    #p1 = np.r_[x1, shape_function(*x1)]
    #p1_image_point = np.r_[x1, -shape_function(*x1)]
    vec_x0x1 = p1-p0
    norm = np.linalg.norm(vec_x0x1)
    if norm == 0:
        norm = 1
    vec_x0x1 = vec_x0x1/norm
    cos = np.dot(vec_x0x1, k_incident)
    dist1 = np.linalg.norm(p0-p1)
    dist2 = np.linalg.norm(p0-p1_image_point)
    #print "vec_x0x1 = ", vec_x0x1
    #print "cos = ", cos
    #print "p0 = ", p0
    #print "p1 = ", p1
    #print "dist1 = ", dist1
    #print "dist2 = ", dist2
    OPD_part1 = dist1*cos*n1
    OPD_part2 = dist2*n2
    OPD = OPD_part2-OPD_part1
    return OPD

@profile
def pattern(opd):
    intensity = 1+np.cos((2*np.pi/0.532)*opd)
    return intensity

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.mlab import griddata
    import numpy as np
    import sys
    import progressbar
    import os
    from itertools import product
    import time
    from colorama import Style, Fore
    start = time.time()
    print "starting..."
    i = 0
    phi = 0
    framenumber =1 
    for theta in np.linspace(0.,0.1,framenumber):
        i += 1
        pltnumber = 100 
        pltlength = 350
        coordinates = np.array(list(product(np.linspace(-pltlength,pltlength,pltnumber), np.linspace(-pltlength, pltlength, pltnumber))))
        q = 0
        intensity = np.zeros((coordinates.shape[0], ))
        for detecting_point in coordinates:
            opd = optical_path_diff(k_incident = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi), -np.cos(theta)]),\
                    x1 = detecting_point,\
                    n1 = 1.5,\
                    n2 = 1)
            intensity[q] = pattern(opd)
            q+=1
            #opd_expected = 2*shape_function(0)*np.cos(np.arcsin(np.sin(angle-0.0000001)*1.5)+0.0000001)
            #print pattern(opd)
        #print "error in OPD = " ,(opd-opd_expected)/0.532, "wavelength"
        X = coordinates[:,0].reshape((pltnumber,pltnumber))
        Y = coordinates[:,1].reshape((pltnumber,pltnumber))
        Z = intensity.reshape((pltnumber, pltnumber))
        fig = plt.figure(num=None, figsize=(8, 7), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('$x,\mu m$')
        ax.set_ylabel('$y,\mu m$')
        ax.set_zlim(0,4)
        ax.set_zticks([0,2,4])
        ax.plot_wireframe(X,Y,Z)
        ax.elev = 80
        ax.azim = -63
        #ax.text(0, 2.2, r'$rotated : %.4f rad$'%theta, fontsize=15)
        dirname = "./movie2D3/"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(dirname+'{:4.0f}'.format(i)+'.tif')
        plt.close()
        progressbar.progressbar_tty(i, framenumber, 1)
    print "finished!"
    print(Fore.CYAN)
    print "Total running time:", time.time()-start, 'seconds'
    print(Style.RESET_ALL)
