from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap('tab10')
x = np.arange(0,20, 0.001)
red = 1+np.cos(4*np.pi*(x+0.630/4)/0.630)
amber = 1+ np.cos(4*np.pi*(x+0.59/4)/0.590)
#plt.plot(x, red+amber)
plt.title('red and amber')
plt.plot(x, red,color=cmap(3))
plt.plot(x, amber, color=cmap(1))
plt.show()

