from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
from plotwithsliders import plotwithsliders as ps
from plotwithsliders import sliders_buttons as sb 
from find_peaks import find_indices_max as fimax
from find_peaks import find_indices_min as fimin
cmap = plt.get_cmap('tab10')
am = cmap(1)
gr = cmap(2)
rd = cmap(3)
x = np.arange(0,20, 0.001)
red = 1+np.cos(4*np.pi*(x+0.630/4)/0.630)
amber = 1+ np.cos(4*np.pi*(x+0.59/4)/0.590)
fig,ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_ylim(-1,3)
for i,ind in enumerate(fimin(amber)):
    ax.annotate('%d'%(i+1),xy=(x[ind],0),xytext=(x[ind],-0.1),color=am)
for i,ind in enumerate(fimin(red)):
    ax.annotate('%d'%(i+1),xy=(x[ind],0),xytext=(x[ind],-0.2),color=rd)
pararange = [0.5,0.6]
parainit = 0.532
slider,buttonplus,buttonminus = sb(pararange,parainit)
ax.plot(x, red, color=rd)
ax.plot(x, amber, color=am)
def xgreen(wvlg):
    return x 
def ygreen(wvlg):
    return 1+ np.cos(4*np.pi*(xgreen(wvlg)+wvlg/4)/wvlg)
ps([slider],[buttonplus],[buttonminus],ax,xgreen,ygreen,gr,[pararange],[parainit])
#plt.title('green and amber')
plt.show()

