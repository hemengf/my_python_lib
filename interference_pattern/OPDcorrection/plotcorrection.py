from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
theta = np.arange(0,0.02,0.001)
n1 = 1.5
n2 = 1
a1=  np.pi/2
OB =500*1000 
a2 = np.arccos((n2/n1)*np.sin(np.arcsin((n1/n2)*np.cos(a1)+2*theta)))
s = (np.sin((a1-a2)/2))**2
dL = -2*n1*OB*s
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
l, = plt.plot(theta,dL)
ax.set_ylim(-600,600)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('nm')


xa1slider = plt.axes([0.25,0.02,0.65,0.03])
xOBslider = plt.axes([0.25,0.05,0.65,0.03])
a1slider = Slider(xa1slider,'a1',np.pi/2-0.5,np.pi/2,valinit=np.pi/2-0.5)
OBslider = Slider(xOBslider,'OB',-500,1000,valinit=0)
def update(val):
    OB = OBslider.val*1000
    a1 = a1slider.val
    a2 = np.arccos((n2/n1)*np.sin(np.arcsin((n1/n2)*np.cos(a1)+2*theta)))
    s = (np.sin((a1-a2)/2))**2
    dL = -2*n1*OB*s
    #fig.canvas.draw_idle()
    l.set_ydata(dL)
    ax.set_ylim(-600,600)
a1slider.on_changed(update)
OBslider.on_changed(update)


plt.show()

