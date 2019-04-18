from __future__ import division
import scipy.optimize
import scipy.spatial.distance
#from scipy.misc import derivative
import partial_derivative
import math
import sys

#@profile
def shape_function(x):
    #return np.exp(-0.00002*((x+250)**2)) 
    #return -0.000008*(x**2)+ float(sys.argv[1])
    return 0.00000001*x + float(sys.argv[1])
    #return 0.00000001*x +68.362


#@profile
def find_k_refracting(k_incident, x1, n1,n2):
    gradient = partial_derivative.derivative(shape_function, x1, dx=1e-6) 
    n = np.empty((2,))
    n[0] = -gradient
    n[1] = 1
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

#@profile
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
        return k_refracting[0]*(shape_function(x1)+shape_function(x))+k_refracting[1]*(x1-x)
    #x0 = scipy.optimize.newton_krylov(F,x1,f_tol = 1e-3) 
    x0 = scipy.optimize.root(F,x1)
    x0 = x0.x[0]
    return x0

#@profile
def optical_path_diff(k_incident, x1, n1,n2):
    x0 = find_x0(k_incident, x1, n1, n2)
    p0 = np.empty((2,))
    p1 = np.empty((2,))
    p1_image_point = np.empty((2,))
    p0[0] = x0
    p1[0] = x1
    p1_image_point[0] = x1
    p0[1] = shape_function(x0)
    p1[1] = shape_function(x1)
    p1_image_point[1] = -shape_function(x1)
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

#@profile
def pattern(opd):
    intensity = 1+np.cos((2*np.pi/0.532)*opd)
    return intensity

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.mlab import griddata
    import numpy as np
    import progressbar
    import os
    from itertools import product
    import time
    from colorama import Style, Fore
    import find_center
    import cookb_signalsmooth
    start = time.time()
    print "starting..."
    i = 0
    framenumber = 50  
    pltnumber = 300
    pltlength = 500
    center = 0
    center_array = np.empty((framenumber, ))
    coordinates = np.linspace(-pltlength, pltlength, pltnumber)
    intensity = np.empty((pltnumber, ))
    intensity2 = np.empty((pltnumber, ))
    for theta in np.linspace(0.,0.0416,framenumber):
        i += 1
        #coordinates = np.array(list(product(np.linspace(-pltlength,pltlength,pltnumber), np.linspace(-pltlength, pltlength, pltnumber))))
        q = 0
        for detecting_point in coordinates:
            opd = optical_path_diff(k_incident = np.array([np.sin(theta), -np.cos(theta)]),\
                    x1 = detecting_point,\
                    n1 = 1.5,\
                    n2 = 1)
            intensity[q] = pattern(opd)

            opd2= 2*68.362*np.cos(np.arcsin(1.5*np.sin(theta)))# from simple formula 2nhcos(j) for air gap for sanity check; should be close
            intensity2[q] = pattern(opd2)

            q+=1
            #opd_expected = 2*shape_function(0)*np.cos(np.arcsin(np.sin(angle-0.0000001)*1.5)+0.0000001)
            #print pattern(opd)
        #print "error in OPD = " ,(opd-opd_expected)/0.532, "wavelength"
        #fig = plt.figure(num=None, figsize=(8, 7), dpi=100, facecolor='w', edgecolor='k')
        #np.save('intensity.npy', intensity)
        #intensity_smooth = cookb_signalsmooth.smooth(intensity, 15)
        #xcenter = find_center.center_position(intensity, coordinates,center)
        #center = xcenter
        #plt.plot(coordinates,intensity_smooth)
        #plt.plot(coordinates,intensity)
        #plt.show()
        #center_array[i-1] = center 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x,\mu m$')
        ax.set_ylim(0,2.5)
        ax.plot(coordinates, intensity)
        #ax.plot(coordinates[int(len(coordinates)/2):], intensity2[int(len(coordinates)/2):],'r') #for sanity check
        ax.text(0, 2.2, r'$rotated : %.4f rad$'%theta, fontsize=15)
        dirname = "./movie/"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(dirname+'{:4.0f}'.format(i)+'.tif')
        plt.close()
        progressbar.progressbar_tty(i, framenumber, 1)
    if not os.path.exists("./output_test"):
        os.makedirs("./output_test")
    #np.save("./output_test/center_array_%d.npy"%int(sys.argv[1]), center_array)
    print(Fore.CYAN)
    print "Total running time:", time.time()-start, "seconds"
    print(Style.RESET_ALL)
    print "center height:", sys.argv[1]
    print "Finished!"
    #plt.plot(np.linspace(0,0.06, framenumber), center_array)
    #plt.show()

