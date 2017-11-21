from __future__ import division
import numpy as np
import warnings

def exact_local_maxima1D(a): 
    """
    Compare adjacent elements of a 1D array.

    Returns a np array of true values for each element not counting
    the first and last element.
    Modified from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    
    """

    true_values = np.greater(a[1:-1], a[:-2]) & np.greater(a[1:-1], a[2:])
    return true_values

def exact_local_minima1D(a):
    true_values = np.less(a[1:-1], a[:-2]) & np.less(a[1:-1], a[2:])
    return true_values

def right_edge_local_maxima1D(a):
    """
    For the case of plateaus coexisting with peaks.

    Returns a boolean array excluding the first and last
    elements of the input array.
    In case of a plateau, the right edge is considered 
    a peak position.
    """

    warnings.filterwarnings("ignore")
    aa = np.copy(a) # make sure input itself won't be modified
    diff= np.diff(aa)
    smallest_diff = np.min(abs(diff[np.nonzero(diff)]))
    aa[diff==0.] -= smallest_diff/2
    true_values = np.greater(aa[1:-1], aa[:-2]) & np.greater(aa[1:-1], aa[2:])
    return true_values

def left_edge_local_maxima1D(a):
    """
    Similar to right_edge_local_maxima2D().
    """
    aa = a.copy()
    diff = np.diff(aa)
    diff = np.insert(diff, 0, 1)
    smallest_diff = np.min(abs(diff[np.nonzero(diff)]))
    aa[diff==0.] -= smallest_diff/2
    true_values = np.greater(aa[1:-1], aa[:-2]) & np.greater(aa[1:-1], aa[2:])
    return true_values

def right_edge_local_minima1D(a):
    """
    Similar to right_edge_local_maxima1D().
    """

    warnings.filterwarnings("ignore")
    aa = np.copy(a) # make sure input itself won't be modified
    diff= np.diff(aa)
    smallest_diff = np.min(abs(diff[np.nonzero(diff)]))
    aa[diff==0.] += smallest_diff/2
    true_values = np.less(aa[1:-1], aa[:-2]) & np.less(aa[1:-1], aa[2:])
    return true_values

def left_edge_local_minima1D(a):
    """
    Similar to right_edge_local_minima2D().
    """
    aa = a.copy()
    diff = np.diff(aa)
    diff = np.insert(diff, 0, 1)
    smallest_diff = np.min(abs(diff[np.nonzero(diff)]))
    aa[diff==0.] += smallest_diff/2
    true_values = np.less(aa[1:-1], aa[:-2]) & np.less(aa[1:-1], aa[2:])
    return true_values

def find_indices_max(a):
    """
    Find indices of local maxima.
    Returns a np array of indices.
    """

    true_values = exact_local_maxima1D(a)
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices

def find_indices_min(a):
    true_values = exact_local_minima1D(a)
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices

def find_indices_all(a):
    """
    Find indices of all local extrema.
    Returns a np array of indices.
    """

    true_values_max = exact_local_maxima1D(a)
    true_values_min = exact_local_minima1D(a)
    true_values = true_values_max | true_values_min
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices
    
def left_find_indices_max(a):
    true_values = left_edge_local_maxima1D(a)
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices

def left_find_indices_min(a):
    true_values = left_edge_local_minima1D(a)
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices

def right_find_indices_max(a):
    true_values = right_edge_local_maxima1D(a)
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices

def right_find_indices_min(a):
    true_values = right_edge_local_minima1D(a)
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices

def left_find_indices_all(a):
    true_values_max = left_edge_local_maxima1D(a)
    true_values_min = left_edge_local_minima1D(a)
    true_values = true_values_max | true_values_min
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices

def right_find_indices_all(a):
    true_values_max = right_edge_local_maxima1D(a)
    true_values_min = right_edge_local_minima1D(a)
    true_values = true_values_max | true_values_min
    indices = [i for i,x in enumerate(true_values) if x== True]
    indices = np.array(indices) + 1
    return indices


if __name__ == "__main__":
    a = np.array([2,3,1,2,3,2,1,2,3,2,1,2,3,2])
    s = exact_local_minima1D(a)
    s1 = find_indices_min(a)
    s2 = find_indices_max(a)
    s3 = find_indices_all(a)
    b = np.array([-1,4,4,2,3,3,3,3,2,6,1])
    b = b.astype("float")
    print "if minima(not counting the first the last element)", s, type(s)
    print "min indices:", s1, type(s1)
    print "max indices:", s2, type(s2)
    print "all peaks:", s3, type(s3)
    print left_find_indices_all(b)
    print b


