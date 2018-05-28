from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
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
#lred,= ax.plot(x, red, color=rd, visible=False)
lamber, = ax.plot(x, amber, color=am,visible=False)
for i,ind in enumerate(fimin(amber)):
    ax.annotate('%d'%(i+1),xy=(x[ind],0),xytext=(x[ind],-0.1),color=am)
for i,ind in enumerate(fimin(red)):
    ax.annotate('%d'%(i+1),xy=(x[ind],0),xytext=(x[ind],-0.2),color=rd)
pararange = [0.5,0.6]
parainit = 0.532
slider,buttonplus,buttonminus = sb(pararange,parainit)
def xgreen(wvlg):
    return x 
def ygreen(wvlg):
    return 1+ np.cos(4*np.pi*(xgreen(wvlg)+wvlg/4)/wvlg)
lgreen = ps([slider],[buttonplus],[buttonminus],ax,xgreen,ygreen,gr,[pararange],[parainit])

parainitred = 0.630
pararangered = [0.6,0.7]
sliderred,buttonplusred,buttonminusred = sb(pararangered,parainitred, height=0.12)
def xred(wvlg):
    return x 
def yred(wvlg):
    return 1+ np.cos(4*np.pi*(xred(wvlg)+wvlg/4)/wvlg)
lred = ps([sliderred],[buttonplusred],[buttonminusred],ax,xred,yred,rd,[pararangered],[parainitred])





rax = plt.axes([0.01, 0.4, 0.1, 0.15])
check = CheckButtons(rax, ('red', 'amber', 'green'), (True, False, True))
def func(label):
    if label == 'red':
        lred.set_visible(not lred.get_visible())
    elif label == 'amber':
        lamber.set_visible(not lamber.get_visible())
    elif label == 'green':
        lgreen.set_visible(not lgreen.get_visible())
    plt.draw()
check.on_clicked(func)
plt.show()

