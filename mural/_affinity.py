import numpy as np
from collections import deque


def binary_affinity(result_list):
    """
    Calculates the average binary affinity from a list of results of each tree.
    
    Arguments:
    result_list -- a list of "classification" results for a given sample on each tree
    
    Return:
    the elementwise average of binary affinities across all the trees
    """

    n = result_list[0].shape[0] # We have this many observations
    M_acc = np.zeros(shape=(n,n))

    for idx in result_list:
        I1 = np.tile(idx, [n,1])
        I2 = np.tile(idx[:,None], [1,n])
        # Force broadcasting to get a binary affinity matrix
        # (Do observations i and j have the same leaf assignment?)
        M_acc += np.equal(I1, I2).astype("int")

    return M_acc / len(result_list)


def adjacency_to_distances(Al, Ll=None, weighted=True, geometric=False):
    """
    Takes adjacency list and returns a distance matrix on its decision tree. 
    Manually runs breadth-first search to avoid calling networkx methods 
    and getting a dictionary intermediate.
    
    Arguments:
    Al -- an adjacency list of a tree
    Ll -- a list of leaves of a tree (optional)
    weighted -- True (default) or False
    geometric -- True or False (default)
    
    Return:
    a distance matrix on the tree
    """

    # The number of nodes in one decision tree
    n = len(Al)

    if Ll is None:
        Ll = range(0, n)

    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)

    # The steps below compute shortest paths by calling BFS from each node
    for i in Ll:
        # Initialize a queue and push the starting point onto it
        q = deque()
        q.append(i)

        # Keep track of what nodes we visited in this search
        visited = np.zeros(n)
        # We should not come back to where we started
        visited[i] = 1

        # Keep searching until we run out of unseen nodes
        while q:
            # Look at the node we discovered earliest
            curr = q.popleft()

            # Check all neighbors
            for j, w in Al[curr]:
                # To see if they were not seen before
                if not visited[j]:
                    # Then mark the node as visited
                    visited[j] = 1

                    # j's distance from node i is w more than that from i to j's predecessor (curr)
                    if ((not weighted) and w != 0):
                        ww = 1
                    else:
                        ww = w

                    dist[i][j] = dist[i][curr] + ww

                    # Add j to the queue so that we can visit its neighbors later
                    q.append(j)

    # The distance matrix for this tree is now done

    if geometric:
        mask = np.nonzero(dist == 0) # Pairs of arguments in the same leaves
        dist = 2 ** (dist - 1) # Reweight paths
        dist[mask] = 0 # But keep zero distances

    return dist


def get_average_distance(D_list, result_list):
    """
    Computes the average distance between two observations across all 
    decision trees in a forest.
    
    Arguments:
    D_list -- a list containing the distance matrices for each tree in the given forest
    result_list -- already found list of results of each tree
    
    Return:
    the matrix of average distances for the observations in this batch
    """

    assert len(D_list) == len(result_list)

    n = result_list[0].shape[0]
    M_acc = np.zeros(shape=(n,n))

    for D, idx in zip(D_list, result_list):
        # Choose M(x_i, x_j) to be the distance from leaf(x_i) to leaf(x_j)
        M_acc += D[idx][:,idx]

    return M_acc / len(result_list)
