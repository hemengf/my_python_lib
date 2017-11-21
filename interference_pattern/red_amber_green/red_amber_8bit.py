from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,20, 0.001)
red = 1+np.cos(4*np.pi*(x+0.630/4)/0.630)
amber = 1+ np.cos(4*np.pi*(x+0*0.59/4)/0.590)
plt.plot(x, red+amber)
plt.title('red and amber 8bit')
plt.plot(x, red, 'r')
plt.plot(x, amber, 'y')
plt.show()

