import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('data_lambda1vsangle')
lambda1 = 0.5*(data[:,2]+data[:,3])
angle = 0.5*(180-data[:,0]+data[:,1])*np.pi/180.
cosangle = np.cos(angle)
sinangle = np.sin(angle)
anglefunction = sinangle/np.power(cosangle,0.33)
plt.scatter(anglefunction, lambda1, s=30, facecolors='none',edgecolors='k')
plt.axis([0,1.5,0,160])
plt.xlabel(r'$\frac{\sin\phi}{\cos^\frac{1}{3}\phi}$',fontsize=20)
plt.ylabel(r'$\lambda_1$',fontsize=20)
plt.gcf().subplots_adjust(bottom = 0.15)
plt.savefig('lambdavsangle.png')
plt.show()
