from __future__ import division
import find_peaks
import numpy as np
def center_position(intensity, x, center):
    left_indices = find_peaks.left_find_indices_all(intensity)
    left_x_position = x[left_indices]
    left_center_idx = np.abs(left_x_position-center).argmin()
    right_indices = find_peaks.right_find_indices_all(intensity)
    right_x_position = x[right_indices]
    right_center_idx = np.abs(right_x_position-center).argmin()
    return (left_x_position[left_center_idx]+right_x_position[right_center_idx])/2


if __name__ == "__main__":
    from scipy import signal
    import matplotlib.pyplot as plt
    intensity = np.load('intensity.npy')
    coordinates = np.linspace(-500,500,300)
    peak = center_position(intensity,coordinates, 0)
    plt.plot(coordinates, intensity)
    plt.axvline(x = peak)
    plt.show()
