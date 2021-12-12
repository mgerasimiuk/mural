import numpy as np
import pickle
import json


def adjacency_matrix_from_list(Al):
    """
    Makes an adjacency matrix from an adjacency list representing the same graph
    
    Arguments:
    Al -- an adjacency list (Python type)
    
    Return:
    an adjacency matrix (ndarray)
    """

    n = len(Al)
    Am = np.zeros((n, n), dtype=int)

    for i in range(n):
        for v, w in Al[i]:
            Am[i, v] = w

    return Am


def load_pickle(path):
    """
    Loads an UnsupervisedForest object from pickle.
    
    Arguments:
    path -- path to a saved UnsupervisedForest object
    
    Return:
    the UnsupervisedForest object
    """

    f = open(path, "rb")
    result = pickle.load(f)
    f.close()
    return result


def load_json(path):
    """
    Loads an UnsupervisedForest object from json file.
    
    Arguments:
    path -- path to a saved UnsupervisedForest object
    
    Return:
    the UnsupervisedForest object
    """

    f = open(path, "r")
    result = json.load(f)
    f.close()
    return result
