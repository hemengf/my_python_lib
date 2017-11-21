from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
Rmin = 1
Rmax = 5
R = np.arange(Rmin,Rmax,0.01)
for U in np.arange(0.09,0.136,0.01):
    v = 438*1e-6
    rhs = np.sqrt(1e6*v*U/9.8)*np.sqrt(2/np.log(7.4*v/(2*R*1e-3*U)))
    plt.plot(R, rhs)
plt.plot(R, R)
plt.ylim(Rmin,Rmax)
plt.ylim(Rmin,Rmax)
Re = 6*1e-3*0.1/v
print 'Re = ', Re
plt.show()
