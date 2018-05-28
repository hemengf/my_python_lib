import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button 
def sliders_buttons(pararange,parainit,height = 0.08,incremental=0.001):
    xslider = plt.axes([0.25,height,0.65,0.03])
    slider = Slider(xslider,'para',pararange[0],pararange[1],valinit=parainit,valfmt='%1.3f')
    xbuttonminus= plt.axes([0.1,height,0.02,0.03])
    xbuttonplus= plt.axes([0.12,height,0.02,0.03])
    buttonplus = Button(xbuttonplus,'+')
    buttonminus = Button(xbuttonminus,'-')
    def incr_slider(val):
        slider.set_val(slider.val+incremental)
    def decr_slider(val):
        slider.set_val(slider.val-incremental)
    buttonplus.on_clicked(incr_slider)
    buttonminus.on_clicked(decr_slider)
    return slider,buttonplus,buttonminus

def plotwithsliders(slider,buttonplus,buttonminus,ax,x,y,mycolor,pararange,parainit):
    para = parainit
    lines, = ax.plot(x(*para),y(*para),color=mycolor) 

    def update(arbitrary_arg):
        for i in range(len(slider)):
            para[i] = slider[i].val
        lines.set_xdata(x(*para))
        lines.set_ydata(y(*para))
        plt.draw()
        #fig.canvas.draw_idle()
    for i in range(len(slider)):
        slider[i].on_changed(update)
    return lines

