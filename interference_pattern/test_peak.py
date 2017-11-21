import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cookb_signalsmooth

intensity = np.load("intensity.npy")
intensity = -intensity
coordinates = np.linspace(-500,500,300)
plt.plot(coordinates, intensity)
#intensity = cookb_signalsmooth.smooth(intensity, 10)
#plt.plot(coordinates, intensity)
peakind = signal.find_peaks_cwt(intensity, np.arange(20,150))
plt.plot(coordinates[peakind], intensity[peakind],'+', color = 'r')
plt.show()

