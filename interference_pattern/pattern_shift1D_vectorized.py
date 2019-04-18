from __future__ import division
from scipy.misc import derivative
import scipy.optimize
import scipy.spatial.distance

def shape_function(x):
    return 0.000005*(x**2)+68
    #return 0.00000001*x + 68

#@profile
def find_k_refracting(k_incident, x1, n1,n2):
    #n = np.array([[-derivative(shape_function, x, dx=1e-6), 1]for x in x1])
     #above method in creating n is too slow
    n = np.empty((len(x1), 2))
    n[:,0] = -derivative(shape_function, x1, dx=1e-6)
    #n[:,0] = -partial_derivative.derivative(shape_function, x1, dx=1e-6)
    n[:,1] = 1
    norm = np.linalg.norm(n, axis = 1)
    n = n/norm[:,np.newaxis]
    c = -np.dot(n, k_incident)
    r = n1/n2
    if ((1-r**2*(1-c**2)) < 0).any():
        print(Fore.RED)
        print "Total internal reflection occurred."
        print "1-r**2*(1-c**2) = \n", 1-r**2*(1-c**2)
        print(Style.RESET_ALL)
        sys.exit(0)
    factor = (r*c- np.sqrt(1-r**2*(1-c**2)))
    #print "n = ", n
    #print 'c =',c 
    #print "factor", factor 
    #print "tile", np.tile(r*k_incident,(len(x1), 1))
    #print k_refracting
    k_refracting = np.tile(r*k_incident,(len(x1), 1)) + n*factor[:,np.newaxis]
    return k_refracting

#@profile
def find_x0(k_incident, x1, n1,n2):
    #def g(x):
    #    k_refracting = find_k_refracting(k_incident, x, n1, n2)
    #    #return -k_refracting[:,1]/k_refracting[:,0]
    #    return k_refracting[:,0], k_refracting[:,1]
    def F(x):
        k_refracting = find_k_refracting(k_incident, x, n1, n2)
        #return shape_function(x1)+shape_function(x)-(x1-x)*g(x)
        return k_refracting[:,0]*(shape_function(x1)+shape_function(x))+k_refracting[:,1]*(x1-x)
    x0 = scipy.optimize.newton_krylov(F,x1, f_tol = 1e-3) 
    return x0

#@profile
def optical_path_diff(k_incident, x1, n1,n2):
    x0 = find_x0(k_incident, x1, n1, n2)
    p0 = np.empty((len(x1),2))
    p1 = np.empty((len(x1),2))
    p1_image_point = np.empty((len(x1),2))
    p0[:,0] = x0
    p0[:,1] = shape_function(x0)
    p1[:,0] = x1
    p1[:,1] = shape_function(x1)
    p1_image_point[:,0] = x1
    p1_image_point[:,1] = -shape_function(x1)
    #p0 = np.array([x0, shape_function(x0)])
    #p1 = np.array([x1, shape_function(x1)])
    #p1_image_point = np.array([x1, -shape_function(x1)])
    vec_x0x1 = p1-p0
    norm = np.linalg.norm(vec_x0x1, axis = 1)
    norm[norm == 0] = 1
    vec_x0x1 = vec_x0x1/norm[:,np.newaxis]

    cos = np.dot(vec_x0x1, k_incident)
    dist1 = np.linalg.norm(p0-p1, axis = 1)
    dist2 = np.linalg.norm(p0-p1_image_point, axis = 1)
    #dist1 = scipy.spatial.distance.cdist(p0.T,p1.T,'euclidean')
    #dist2 = scipy.spatial.distance.cdist(p0.T,p1_image_point.T,'euclidean')
    #dist1 = np.diagonal(dist1)
    #dist2 = np.diagonal(dist2)
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

def pattern(opd):
    intensity = 1+np.cos((2*np.pi/0.532)*opd)
    return intensity

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    import progressbar
    import os
    import time
    from colorama import Fore, Style
    start = time.time()
    print "starting..."
    i = 0
    framenumber = 50
    pltnumber = 300
    pltlength = 500
    detecting_range = np.linspace(-pltlength,pltlength,pltnumber)
    for angle in np.linspace(0,0.0625,framenumber):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        i += 1
        opd = optical_path_diff(k_incident = np.array([np.sin(angle),-np.cos(angle)]),\
                x1 = detecting_range,\
                n1 = 1.5,\
                n2 = 1)
        intensity = pattern(opd)
        #opd_expected = 2*shape_function(0)*np.cos(np.arcsin(np.sin(angle-0.00000001)*1.5)+0.00000001)
        #print "error in OPD = " ,(opd-opd_expected)/0.532, "wavelength"
        ax.plot(detecting_range, intensity)
        plt.ylim((0,2.5))
        ax.set_xlabel('$\mu m$')
        ax.text(0, 2.2, r'$rotated : %.4f rad$'%angle, fontsize=15)
        dirname = "./movie/"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(dirname+'{:4.0f}'.format(i)+'.tif')
        plt.close()
        progressbar.progressbar_tty(i, framenumber, 1)
    print(Fore.CYAN)
    print "Total running time:", time.time()-start, "seconds"
    print(Style.RESET_ALL)
    print "finished!"
