#!/usr/bin/env python
import cookb_signalsmooth
import numpy as np
import matplotlib.pyplot as plt
import sys
from find_peaks import exact_local_maxima1D, exact_local_minima1D

def stripes_counting(datafile_name):
    """
    Given a 1-D array of grayscale data, find the peak number
    and the valley number.
    Data could be obtained by imagej grayscale measurement.
    """

    pixel_values = np.loadtxt(datafile_name, skiprows = 1)
    window_len = 10
    smooth_values = cookb_signalsmooth.smooth(pixel_values[:,1], window_len)
    plt.plot(smooth_values)
    plt.plot(pixel_values[:,1])
    plt.show()
    s = raw_input("Is this smoothing (window_len = %d) good enough? (y/n)"%window_len)
    sys.stdout.flush()
    if s == "n":
        unsatisfied = 1
        while unsatisfied:
            t = raw_input("Keep adjusting window length. New window_len = ")
            window_len = int(t)
            smooth_values = cookb_signalsmooth.smooth(pixel_values[:,1], window_len)
            plt.plot(smooth_values)
            plt.plot(pixel_values[:,1])
            plt.show()
            u = raw_input("Is this smoothing (window_len = %d) good enough? (y/n)"%window_len)
            if u=="y":
                true_values_maxima = exact_local_maxima1D(smooth_values)
                maxima_number = np.sum(true_values_maxima)
                true_values_minima = exact_local_minima1D(smooth_values)
                minima_number = np.sum(true_values_minima)
                break

    elif s == "y":
        true_values_maxima = exact_local_maxima1D(smooth_values)
        maxima_number = np.sum(true_values_maxima)
        true_values_minima = exact_local_minima1D(smooth_values)
        minima_number = np.sum(true_values_minima)
    else:
        print "You didn't press anything..."
    return maxima_number, minima_number

if __name__ == "__main__":
    import os
    import sys
    s = ""
    while not os.path.exists(s+".xls"):
        s = raw_input("Give me a correct data file name: ")
        sys.stdout.flush()
    maxima_number, minima_number = stripes_counting(s + ".xls")
    print "%d maxima"%maxima_number
    print "%d minima"%minima_number
    raw_input('press enter')
