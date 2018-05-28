from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from find_peaks import find_indices_max as fimax
from find_peaks import find_indices_min as fimin
cmap = plt.get_cmap('tab10')
am = cmap(1)
gr = cmap(2)
rd = cmap(3)
x = np.arange(0,30, 0.0009)
red = 1+np.cos(4*np.pi*(x+0.630/4)/0.630)
amber = 1+ np.cos(4*np.pi*(x+0.590/4)/0.590)
green = 1+ np.cos(4*np.pi*(x+0.532/4)/0.532)
red8 = 1+np.cos(4*np.pi*x/0.630)
amber8 = 1+ np.cos(4*np.pi*x/0.590)
green8 = 1+ np.cos(4*np.pi*x/0.532)
fig,ax= plt.subplots()
#for i,ind in enumerate(fimin(amber)):
#    ax.annotate('%d'%(i+1),xy=(x[ind],0),xytext=(x[ind],-0.1),color=am)
for i,ind in enumerate(fimin(red)):
    ax.annotate('%d'%(i+1),xy=(x[ind],0),xytext=(x[ind],-0.2),color=rd)
    ax.annotate('%.3f'%(x[ind]),xy=(x[ind],0),xytext=(x[ind],-0.3),color=rd)
for i,ind in enumerate(fimax(red)):
    ax.annotate('%.3f'%(x[ind]),xy=(x[ind],0),xytext=(x[ind],2+0.2),color=rd)
plt.subplots_adjust(bottom=0.2)
lred, = ax.plot(x, red,color=rd,visible=False)
lamber, = ax.plot(x, amber, color=am,visible=False)
lgreen, = ax.plot(x, green, color=gr,visible=False)
lred8, = ax.plot(x, red8,color=rd,visible=False)
lamber8, = ax.plot(x, amber8, color=am,visible=False)
lgreen8, = ax.plot(x, green8, color=gr,visible=False)
#ax.plot(x,amber+green+red)

rax = plt.axes([0.01, 0.4, 0.1, 0.15])
check = CheckButtons(rax, ('red', 'amber', 'green','red8','amber8','green8'), (False, False, False, False, False, False))


def func(label):
    if label == 'red':
        lred.set_visible(not lred.get_visible())
    elif label == 'amber':
        lamber.set_visible(not lamber.get_visible())
    elif label == 'green':
        lgreen.set_visible(not lgreen.get_visible())
    if label == 'red8':
        lred8.set_visible(not lred8.get_visible())
    elif label == 'amber8':
        lamber8.set_visible(not lamber8.get_visible())
    elif label == 'green8':
        lgreen8.set_visible(not lgreen8.get_visible())
    plt.draw()
check.on_clicked(func)

plt.show()

