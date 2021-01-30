import numpy as np
from sklearn.neighbors import kneighbors_graph

EPSILON = np.finfo(float).eps


def H_one(col):
    """
    Get the single-variable entropy of a set.

    @param col a column of data
    @return the Shannon entropy of the set
    """

    # Calculate the optimal numbers of bins for the histograms
    bins = np.histogram_bin_edges(col, bins="auto")

    # Estimate the distributions on each side of the split
    dist = np.histogram(col, bins=bins, density=True)[0]
            
    # Calculate Shannon entropy of the resulting distributions
    return -1 * np.sum(dist * np.log(dist + EPSILON))


def H_spectral(data, num_neighbors):
    """
    Get the spectral entropy of a set.

    @param data a dataset
    @param num_neighbors the number of neighbors
    @return the von Neumann entropy of the set
    """

    A = get_nn_graph(data, num_neighbors)
    return get_spectral_entropy(A)


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
