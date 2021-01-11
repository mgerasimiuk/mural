#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np
cimport cython
from numpy.math cimport INFINITY

DTYPE = np.float64

def find_threshold_index(X_sorted, start, end, H):

    assert X_sorted.dtype == DTYPE
    max_x = X_sorted.shape[0]
    assert start >= 0 and end < max_x

    cdef double[:] X_sorted_view = X_sorted

    result = find_threshold_index_helper(X_sorted_view, max_x, start, end, H)
    return result

cdef int find_threshold_index_helper(double[:] X_sorted_view, double max_x, int start_j, int end_j, double entropy):

    cdef double best_score = INFINITY
    cdef int best_threshold = start_j

    cdef double prop

    cdef int[:] bins_low, bins_high
    cdef double[:] dist_low, dist_high
    cdef double H_low, H_high
    
    cdef double score

    cdef int j
    for j in range(start_j, end_j):

        if X_sorted_view[j] == X_sorted_view[j+1]:
            continue
        
        bins_low = np.histogram_bin_edges(X_sorted_view[:j], bins="auto")
        bins_high = np.histogram_bin_edges(X_sorted_view[j:], bins="auto")

        dist_low = np.histogram(X_sorted_view[:j], bins=bins_low, density=True)[0]
        dist_high = np.histogram(X_sorted_view[j:], bins=bins_high, density=True)[0]

        H_low = -1 * np.sum(dist_low * np.log(dist_low))
        H_high = -1 * np.sum(dist_high * np.log(dist_high))

        prop = j / max_x
        score = prop * H_low + (1 - prop) * H_high

        if score < best_score:
            best_score = score
            best_threshold = j
    
    return best_threshold