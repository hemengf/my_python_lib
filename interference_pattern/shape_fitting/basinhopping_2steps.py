#!/usr/bin/env python
from __future__ import division, print_function
import sys
from scipy import interpolate
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from scipy.optimize import basinhopping

def normalize(img_array,normrange):
	#elementmax = np.amax(img_array)
	#elementmin = np.amin(img_array)
	#ratio = (elementmax-elementmin)/normrange
	#normalized_array = (img_array-elementmin)/(ratio+0.00001)
	test = exposure.equalize_hist(img_array)
	return test
	
def difference(reference_img, generated_img, normrange):
	reference_img = normalize(reference_img, normrange)
	generated_img = normalize(generated_img, normrange)
	diff_value = np.sum((reference_img-generated_img)**2)
	return diff_value

def surface_polynomial_1storder(size, max_variation, coeff1storder):
	def poly(x, y):
		poly = max_variation*(coeff1storder[0]*x+coeff1storder[1]*y)
		return poly
	x = np.linspace(0,size[0]-1, size[0])
	y = np.linspace(0,size[1]-1, size[1])
	zz = poly(x[:,None],y[None, :])
	return zz
def nll_1storder(coeff1storder, max_variation, data, normrange):
	#data = normalize(data, normrange)
	height = surface_polynomial_1storder(data.shape, max_variation, coeff1storder)
	#expected = normalize(1+np.cos((2/0.532)*height), normrange)
	expected = 1+np.cos((2/0.532)*height)
	# normalize to [0,1]
	expected /= expected.max()
	return difference(data, expected, normrange)

def surface_polynomial(size, max_variation, coeffhi,coeff1storder):
	def poly(x, y):
		#poly = max_variation*(coeff[0]*x+coeff[1]*y)
		poly = max_variation*(1*coeffhi[0]*x**2+1*coeffhi[1]*y**2+1*coeffhi[2]*x*y+coeff1storder[0]*x+coeff1storder[1]*y+coeffhi[3])
		return poly
	x = np.linspace(0,size[0]-1, size[0])
	y = np.linspace(0,size[1]-1, size[1])
	zz = poly(x[:,None],y[None, :])
	return zz
def nll(coeffhi,coeff1storder, max_variation, data, normrange):
	#data = normalize(data, normrange)
	height = surface_polynomial(data.shape, max_variation, coeffhi,coeff1storder)
	#expected = normalize(1+np.cos((2/0.532)*height), normrange)
	expected = 1+np.cos((2/0.532)*height)
	# normalize to [0,1]
	expected /= expected.max()
	return difference(data, expected, normrange)

if __name__ == "__main__":
	from scipy.optimize import fmin
	import time
	normrange=1

	N = 14 
	sample_size = 15

	t0 = time.time()
	max_variation = 0.012
	reference_intensity = cv2.imread('crop_small.tif', 0)
	reference_intensity = normalize(reference_intensity,1)
	#cv2.imwrite('normalized_crop.tif',255*reference_intensity)
	alist = np.linspace(0,sample_size,N) # x direction
	blist = np.linspace(-sample_size, sample_size,2*N) # y direction
	aa, bb = np.meshgrid(alist,blist)
	diff = np.empty(aa.shape)


	for i in np.arange(alist.size):
		for j in np.arange(blist.size):
			if (j-0.5*len(blist))**2+(i)**2<=(0.*len(alist))**2:
				diff[j,i] = np.nan 
			else:
                                coeff1storder = [aa[j,i],bb[j,i]]
				diff[j,i] = nll_1storder(coeff1storder,max_variation,reference_intensity,1.0)
			sys.stdout.write('\r%i/%i     ' % (i*blist.size+j+1,alist.size*blist.size))
			sys.stdout.flush()
	sys.stdout.write('\n')
	elapsed = time.time() - t0
	print("took %.2f seconds to compute the likelihood" % elapsed)
	index = np.unravel_index(np.nanargmin(diff), diff.shape)
	index = (alist[index[1]], blist[index[0]])
	index = np.array(index)

        initcoeffhi = np.array([[0,0,0,0]])
        coeff1storder = index 
        print(index)
        simplex = 0.1*np.identity(4)+np.tile(initcoeffhi,(4,1))
        simplex = np.concatenate((initcoeffhi,simplex),axis=0)
	#xopt= fmin(nll, initcoeffhi, args = (coeff1storder,max_variation, reference_intensity, normrange))#, initial_simplex=simplex)
	#print(xopt)
	result = basinhopping(nll, initcoeffhi, niter = 4, T=200, stepsize=.1, minimizer_kwargs={'method': 'Nelder-Mead', 'args': (coeff1storder,max_variation, reference_intensity, normrange)}, disp=True)#, callback = lambda x, convergence, _: print('x = ', x))
        xopt = result.x
        print(result.x)
	#fig = plt.figure()
	##plt.contour(aa, bb, diff, 100)
	#ax = fig.add_subplot(111, projection='3d')
	#ax.plot_wireframe(aa,bb,diff)
	#plt.ylabel("coefficient a")
	#plt.xlabel("coefficient b")
	#plt.gca().set_aspect('equal', adjustable = 'box')
	#plt.colorbar()
	#plt.show()
	generated_intensity = normalize(1+np.cos((2/0.532)*surface_polynomial(reference_intensity.shape, max_variation,xopt,coeff1storder)), 1.0)#works for n=1 pocket
	#cv2.imwrite('ideal_pattern.tif', 255*generated_intensity)
	cv2.imshow('', np.concatenate((generated_intensity, reference_intensity), axis = 1))
	cv2.waitKey(0)
	
	#ax = fig.add_subplot(111, projection = '3d')
	#ax.plot_surface(xx[::10,::10], yy[::10,::10], zz[::10,::10])
	#plt.show()
