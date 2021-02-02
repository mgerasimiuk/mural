import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import column_or_1d

EPSILON = np.finfo(float).eps


def H_one(data, obs, var, imputed=None, use_missing=False, num_neighbors=0):
    """
    Get the single-variable entropy of a set.

    @param col a column of data
    @return the Shannon entropy of the set
    """

    col = data[obs, var].reshape(-1)
    mask = np.isnan(col)
    num_missing = np.count_nonzero(mask)

    num_total = len(col) - num_missing * (not use_missing)

    col = col[~mask]

    # Calculate the optimal numbers of bins for the histograms
    bins = np.histogram_bin_edges(col, bins="auto")

    # Estimate the distribution
    dist = np.histogram(col, bins=bins, density=False)[0]
    dist = dist / num_total
    p_missing = num_missing / num_total
            
    # Calculate Shannon entropy of the distribution
    H = -1 * np.sum(dist * np.log(dist + EPSILON))
    if num_missing != 0 and use_missing:
        H -= p_missing * np.log(p_missing + EPSILON)

    return H


def H_two(data, obs, var, imputed=None, use_missing=False, num_neighbors=0):
    """
    Get 2d entropy.
    """

    col1 = data[obs, var[0]].reshape(-1)
    col2 = data[obs, var[1]].reshape(-1)

    mask1 = np.isnan(col1)
    mask2 = np.isnan(col2)
    mask = ~(mask1 | mask2)

    num_missing1 = np.count_nonzero(mask1 &  ~mask2)
    num_missing2 = np.count_nonzero(mask2 &  ~mask1)
    num_missing12 = np.count_nonzero(mask1 & mask2)
    num_missing = np.count_nonzero(mask1 | mask2)

    num_total = len(col1) - num_missing * (~use_missing)

    col1 = col1[mask]
    col2 = col2[mask]

    bins1 = np.histogram_bin_edges(col1, bins="auto")
    bins2 = np.histogram_bin_edges(col2, bins="auto")
    bins = (bins1, bins2)
    #bins = min(len(bins1), len(bins2)) - 1

    dist = np.histogram2d(col1, col2, bins, density=False)[0]
    dist = dist / num_total
    p_missing1 = num_missing1 / num_total
    p_missing2 = num_missing2 / num_total
    p_missing12 = num_missing12 / num_total

    H = -1 * np.sum(dist * np.log(dist + EPSILON))
    if use_missing:
        if num_missing1 != 0:
            H -= p_missing1 * np.log(p_missing1 + EPSILON)
        if num_missing2 != 0:
            H -= p_missing2 * np.log(p_missing2 + EPSILON)
        if num_missing12 != 0:
            H -= p_missing12 * np.log(p_missing12 + EPSILON)

    return H


def H_two_even(data, obs, var, imputed=None, use_missing=False, num_neighbors=0):
    """
    Get 2d entropy with even bins.
    """

    col1 = data[obs, var[0]].reshape(-1)
    col2 = data[obs, var[1]].reshape(-1)

    mask1 = np.isnan(col1)
    mask2 = np.isnan(col2)
    mask = ~(mask1 | mask2)

    num_missing1 = np.count_nonzero(mask1 &  ~mask2)
    num_missing2 = np.count_nonzero(mask2 &  ~mask1)
    num_missing12 = np.count_nonzero(mask1 & mask2)
    num_missing = np.count_nonzero(mask1 | mask2)

    num_total = len(col1) - num_missing * (~use_missing)

    col1 = col1[mask]
    col2 = col2[mask]

    #bins1 = np.histogram_bin_edges(col1, bins="auto")
    #bins2 = np.histogram_bin_edges(col2, bins="auto")
    #bins = np.stack([bins1, bins2])
    #bins = min(len(bins1), len(bins2)) - 1
    bins = 10

    dist = np.histogram2d(col1, col2, bins, density=False)[0]
    dist = dist / num_total
    p_missing1 = num_missing1 / num_total
    p_missing2 = num_missing2 / num_total
    p_missing12 = num_missing12 / num_total

    H = -1 * np.sum(dist * np.log(dist + EPSILON))
    if use_missing:
        if num_missing1 != 0:
            H -= p_missing1 * np.log(p_missing1 + EPSILON)
        if num_missing2 != 0:
            H -= p_missing2 * np.log(p_missing2 + EPSILON)
        if num_missing12 != 0:
            H -= p_missing12 * np.log(p_missing12 + EPSILON)

    return H


def H_three(data, obs, var, imputed=None, use_missing=False, num_neighbors=0):
    """
    Get 3d entropy.
    """

    col1 = data[obs, var[0]].reshape(-1)
    col2 = data[obs, var[1]].reshape(-1)
    col3 = data[obs, var[2]].reshape(-1)

    mask1 = np.isnan(col1)
    mask2 = np.isnan(col2)
    mask3 = np.isnan(col3)
    mask = ~(mask1 | mask2 | mask3)

    num_missing1 = np.count_nonzero(mask1 & (~mask2) & (~mask3))
    num_missing2 = np.count_nonzero(mask2 & (~mask1) & (~mask3))
    num_missing3 = np.count_nonzero(mask3 & (~mask1) & (~mask2))
    num_missing12 = np.count_nonzero(mask1 & mask2 & ~mask3)
    num_missing13 = np.count_nonzero(mask1 & mask3 & ~mask2)
    num_missing23 = np.count_nonzero(mask2 & mask3 & ~mask1)
    num_missing123 = np.count_nonzero(mask1 & mask2 & mask3)
    num_missing = np.count_nonzero(mask1 | mask2 | mask3)

    num_total = len(col1) - num_missing * (~use_missing)

    col1 = col1[mask]
    col2 = col2[mask]
    col3 = col3[mask]

    bins1 = np.histogram_bin_edges(col1, bins="auto")
    bins2 = np.histogram_bin_edges(col2, bins="auto")
    bins3 = np.histogram_bin_edges(col3, bins="auto")
    bins = (bins1, bins2, bins3)
    #bins = min(len(bins1), len(bins2), len(bins3)) - 1

    dist = np.histogramdd((col1, col2, col3), bins, density=False)[0]
    dist = dist / num_total

    p_missing1 = num_missing1 / num_total
    p_missing2 = num_missing2 / num_total
    p_missing3 = num_missing3 / num_total
    p_missing12 = num_missing12 / num_total
    p_missing13 = num_missing13 / num_total
    p_missing23 = num_missing23 / num_total
    p_missing123 = num_missing123 / num_total

    H = -1 * np.sum(dist * np.log(dist + EPSILON))
    if use_missing:
        if num_missing1 != 0:
            H -= p_missing1 * np.log(p_missing1 + EPSILON)
        if num_missing2 != 0:
            H -= p_missing2 * np.log(p_missing2 + EPSILON)
        if num_missing3 != 0:
            H -= p_missing3 * np.log(p_missing3 + EPSILON)
        if num_missing12 != 0:
            H -= p_missing12 * np.log(p_missing12 + EPSILON)
        if num_missing13 != 0:
            H -= p_missing13 * np.log(p_missing13 + EPSILON)
        if num_missing23 != 0:
            H -= p_missing23 * np.log(p_missing23 + EPSILON)
        if num_missing123 != 0:
            H -= p_missing123 * np.log(p_missing123 + EPSILON)

    return H


def H_three_even(data, obs, var, imputed=None, use_missing=False, num_neighbors=0):
    """
    Get 3d entropy with evenly spaced bins.
    """

    col1 = data[obs, var[0]].reshape(-1)
    col2 = data[obs, var[1]].reshape(-1)
    col3 = data[obs, var[2]].reshape(-1)

    mask1 = np.isnan(col1)
    mask2 = np.isnan(col2)
    mask3 = np.isnan(col3)
    mask = ~(mask1 | mask2 | mask3)

    num_missing1 = np.count_nonzero(mask1 & (~mask2) & (~mask3))
    num_missing2 = np.count_nonzero(mask2 & (~mask1) & (~mask3))
    num_missing3 = np.count_nonzero(mask3 & (~mask1) & (~mask2))
    num_missing12 = np.count_nonzero(mask1 & mask2 & ~mask3)
    num_missing13 = np.count_nonzero(mask1 & mask3 & ~mask2)
    num_missing23 = np.count_nonzero(mask2 & mask3 & ~mask1)
    num_missing123 = np.count_nonzero(mask1 & mask2 & mask3)
    num_missing = np.count_nonzero(mask1 | mask2 | mask3)

    num_total = len(col1) - num_missing * (~use_missing)

    col1 = col1[mask]
    col2 = col2[mask]
    col3 = col3[mask]

    #bins1 = np.histogram_bin_edges(col1, bins="auto")
    #bins2 = np.histogram_bin_edges(col2, bins="auto")
    #bins3 = np.histogram_bin_edges(col3, bins="auto")
    #bins = np.stack([bins1, bins2, bins3])
    #bins = min(len(bins1), len(bins2), len(bins3)) - 1
    bins = 5

    dist = np.histogramdd((col1, col2, col3), bins, density=False)[0]
    dist = dist / num_total

    p_missing1 = num_missing1 / num_total
    p_missing2 = num_missing2 / num_total
    p_missing3 = num_missing3 / num_total
    p_missing12 = num_missing12 / num_total
    p_missing13 = num_missing13 / num_total
    p_missing23 = num_missing23 / num_total
    p_missing123 = num_missing123 / num_total

    H = -1 * np.sum(dist * np.log(dist + EPSILON))
    if use_missing:
        if num_missing1 != 0:
            H -= p_missing1 * np.log(p_missing1 + EPSILON)
        if num_missing2 != 0:
            H -= p_missing2 * np.log(p_missing2 + EPSILON)
        if num_missing3 != 0:
            H -= p_missing3 * np.log(p_missing3 + EPSILON)
        if num_missing12 != 0:
            H -= p_missing12 * np.log(p_missing12 + EPSILON)
        if num_missing13 != 0:
            H -= p_missing13 * np.log(p_missing13 + EPSILON)
        if num_missing23 != 0:
            H -= p_missing23 * np.log(p_missing23 + EPSILON)
        if num_missing123 != 0:
            H -= p_missing123 * np.log(p_missing123 + EPSILON)

    return H


def H_many(data, obs, var=None, imputed=None, use_missing=False, num_neighbors=0):
    """
    Sum entropies of many variables.
    """

    n_dims = data.shape[1]
    H = 0

    for d in range(n_dims):
        H += H_one(data, obs, d)

    return H


def H_spectral(data, obs, var=None, imputed=None, use_missing=False, num_neighbors=5):
    """
    Get the spectral entropy of a set.

    @param data a dataset
    @param num_neighbors the number of neighbors
    @return the von Neumann entropy of the set
    """

    if imputed is None:
        raise (ValueError)

    num_neighbors = min(num_neighbors, len(obs) - 1)

    A = get_nn_graph(imputed[obs], num_neighbors)
    H = get_spectral_entropy(A)

    return H


def get_density_matrix(A):
    """
    Take a graph adjacency matrix and return a density matrix
    for the von Neumann entropy calculation.

    @param A the adjacency matrix of a graph
    @return a density matrix
    """

    degree = np.diag(np.sum(A, axis=1))
    L = degree - A # Get the combinatorial graph Laplacian
    rho = L / np.trace(L)

    return rho


def get_spectral_entropy(A):
    """
    Get the spectral entropy of a network from its adjacency matrix.

    @param A an adjacency matrix
    @return the spectral entropy of the network
    """

    rho = get_density_matrix(A)

    w, v = np.linalg.eigh(rho)
    H = - np.sum(w * np.log2(w + EPSILON))

    return H


def get_nn_graph(imputed_data, num_neighbors, weighted=False):
    """
    Get the k nearest neighbor graph of a dataset.

    @param imputed_data a complete dataset
    @param num_neighbors the number of nearest neighbors
    @param weighted True or False
    @return the adjacency matrix of the resulting network
    """
    
    W = kneighbors_graph(imputed_data, num_neighbors).toarray()

    if weighted:
        return W
    else:
        return (W > 0).astype(int)
