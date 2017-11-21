import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
x1,y1 = np.loadtxt('data_center.txt', delimiter=',', unpack = True)
ax.plot(x1, y1, 'x', color = 'r')
x2,y2 = np.loadtxt('data_wall.txt', delimiter=',', unpack=True)
ax.plot(x2, y2, '+', color = 'g')
plt.axis([0,4, 20, 70])
plt.show()
