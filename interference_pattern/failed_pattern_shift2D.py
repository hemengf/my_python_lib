from __future__ import division
import scipy.optimize
import scipy.spatial.distance
import partial_derivative

def shape_function(x,y):
    return 0.000005*(x**2+y**2)+68
    #return 0.00000001*x + 68
def find_k_refracting(k_incident, x1, n1,n2):
    
    #x1 = [[xa,ya],
    #      [xb,yb],
    #      [xc,yc]]
   
    gradient = np.array(partial_derivative.partial_derivative(shape_function, *x1.T))
    #gradient= [[df/dxa,df/dya],
    #           [df/dxb,df/dyb],
    #           [df/dxc,df/dyc]]
    n = np.ones((x1.shape[0], 3))
    n[:,:-1] = gradient
    norm = np.linalg.norm(n, axis = 1)
    n = n/norm[:,np.newaxis]  # n is the unit normal vector pointing 'upward'
    c = -np.dot(n, k_incident)
    r = n1/n2
    if ((1-r**2*(1-c**2)) < 0).any():
        print "Total internal reflection occurred."
        print "1-r**2*(1-c**2) = \n", 1-r**2*(1-c**2)
        sys.exit(0)
    factor = (r*c- np.sqrt(1-r**2*(1-c**2)))
    k_refracting = np.tile(r*k_incident,(x1.shape[0], 1)) + n*factor[:,np.newaxis]
    #print "n = ", n
    #print 'c =',c 
    #print "factor", factor 
    #print "tile", np.tile(r*k_incident,(x1.shape[0], 1))
    #print "k_refracting = ", k_refracting
    return k_refracting

#@profile
def find_x0(k_incident, x1, n1,n2):
    def Fx(x):
        k_refracting = find_k_refracting(k_incident, x, n1, n2)
        return k_refracting[:,0]*(shape_function(*x1.T)+shape_function(*x.T))+k_refracting[:,2]*(x1-x)[:,0]
    def Fy(x):
        k_refracting = find_k_refracting(k_incident, x, n1, n2)
        return k_refracting[:,1]*(shape_function(*x1.T)+shape_function(*x.T))+k_refracting[:,2]*(x1-x)[:,1]
    def F(x):
        return 1e5*(Fx(x)**2 + Fy(x)**2)
    print "F = ", F(x1)
    """
    A FAILED PROJECT.
    
    Having F(x,y,x1,y1) = 0. Easy to root find
    1 pair of x,y given 1 pair of x1,y1. Successful
    in vectorizing F, making it accept a matrix of 
    x1,y1.
    FAILED IN THE NEXT STEP OF ROOT FINDING.
    SCIPY DOESN'T SEEM TO SUPPORT SIMULTANEOUS
    ROOT FINDING (vectorization).
    """

    x0 = scipy.optimize.root(F,x1) 
    return x0

def optical_path_diff(k_incident, x1, n1,n2):
    x0 = find_x0(k_incident, x1, n1, n2)
    p0 = np.concatenate((x0, shape_function(*x0.T)[:,np.newaxis]),axis=1)
    p1 = np.concatenate((x1, shape_function(*x1.T)[:,np.newaxis]),axis=1)
    p1_image_point = np.concatenate((x1, -shape_function(*x1.T)[:,np.newaxis]),axis=1)
    vec_x0x1 = p1-p0
    norm = np.linalg.norm(vec_x0x1, axis = 1)
    norm[norm == 0] = 1
    vec_x0x1 = vec_x0x1/norm[:,np.newaxis]

    cos = np.dot(vec_x0x1, k_incident)
    dist1 = scipy.spatial.distance.cdist(p0,p1,'euclidean')
    dist2 = scipy.spatial.distance.cdist(p0,p1_image_point,'euclidean')
    dist1 = np.diagonal(dist1)
    dist2 = np.diagonal(dist2)
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
    import processbar
    import os
    print "starting..."
    i = 0
    phi = 0
    for theta in np.linspace(0.,0.1,1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        i += 1
        opd = optical_path_diff(k_incident = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi), -np.cos(theta)]),\
                x1 = np.array([[0,10]]),\
                n1 = 1.5,\
                n2 = 1)
        intensity = pattern(opd)
        #opd_expected = 2*shape_function(0)*np.cos(np.arcsin(np.sin(angle-0.0000001)*1.5)+0.0000001)
        print opd
        #print "error in OPD = " ,(opd-opd_expected)/0.532, "wavelength"
        #ax.plot(detecting_range, intensity)
        #plt.ylim((0,2.5))
        #ax.set_xlabel('$\mu m$')
        #ax.text(0, 2.2, r'$rotated : %.4f rad$'%angle, fontsize=15)
        #dirname = "./movie2D/"
        #if not os.path.exists(dirname):
        #    os.makedirs(dirname)
        #plt.savefig(dirname+'{:4.0f}'.format(i)+'.tif')
        #plt.close()
        #processbar.processbar_tty(i, 100, 1)
    print "finished!"
