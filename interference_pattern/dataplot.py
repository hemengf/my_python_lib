import matplotlib.pyplot as plt
import numpy as np
framenumber = 50
fig = plt.figure()
ax = fig.add_subplot(111)
d = {}
height_range = range(0,2000,100)
for i in height_range:
    d["data%d"%i] = np.load("./output_test/center_array_%d.npy"%i)
    d["data%d"%i] = d["data%d"%i][::1]
    angles = np.linspace(0,0.06, framenumber)
    angles = angles[::1]
    plt.plot(angles, d["data%d"%i], 'o-', markersize =i/200)
    ax.set_xlabel("rotated angle, $rad$")
    ax.set_ylabel("center shift $\mu m$")
#plt.plot([q for q in height_range], [d["data%d"%k][-1] for k in height_range])
#ax.set_xlabel("center height, $\mu m$")
#ax.set_ylabel("center shift, $\mu m$")
plt.show()
    
