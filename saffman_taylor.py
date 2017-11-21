from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
w = 236/519 
y = np.arange(-w+0.0005,w-0.0005,0.001)
plt.plot(y, ((1-w)/np.pi)*np.log((1+np.cos(np.pi*y/w))/2))
plt.axes().set_aspect('equal')
plt.xlim(-1,1)
plt.show()
