# Code for testing MURAL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scprep
import phate
from sklearn.manifold import TSNE
from sklearn import preprocessing
from scipy.stats import special_ortho_group
from sklearn.datasets import make_swiss_roll, make_moons
import demap
import os

from base import *
from _entropy import *
from _affinity import *

default_trees = [10, 100, 500]
default_depths = [4, 6, 10]


def graph_vars(df, vars_list):
    """
    Make plots of data distributions.

    @param df a pandas DataFrame
    @param a list of variables to visualize
    """
    df[vars_list].hist(figsize=(16, 20), bins=100, xlabelsize=8, ylabelsize=8)


def make_embedded_swiss_roll(dims=3):
    """
    Make a dataset of the Swiss roll manifold embedded in multiple dimensions.

    @param dims the number of dimensions to embed in
    @return the data matrix for the embedded Swiss roll
    """
    assert dims >= 3

    # Generate Swiss Roll
    x, labels = make_swiss_roll(n_samples=3000, random_state=42)
    if dims > 3:
        x = np.dot(x, special_ortho_group.rvs(dims)[:3])

    # Standardize with mean and standard deviation
    standardized_X = preprocessing.scale(x, with_mean=True, with_std=True)

    return standardized_X, labels


def make_embedded_moons(dims=2):
    """
    Make a dataset of the two moons manifold embedded in multiple dimensions.

    @param dims the number of dimensions to embed in
    @return the data matrix for the embedded moons
    """
    assert dims >= 2

    # Generate moons
    x, labels = make_moons(n_samples=3000, random_state=42)
    x = np.dot(x, special_ortho_group.rvs(dims)[:2])

    # Standardize with mean and standard deviation
    standardized_X = preprocessing.scale(x, with_mean=True, with_std=True)

    return standardized_X, labels

def make_splatter():
    """
    Make a splatter dataset.

    @return the true and noisy data matrices for the splatter dataset
    """
    
    scprep.run.install_bioconductor("splatter")
    data_true = demap.splatter.paths(bcv=0, dropout=0, seed=42)
    data_noisy = demap.splatter.paths(bcv=0.2, dropout=0.5, seed=42)

    return data_true, data_noisy


def make_tree(n_dim=100, n_branch=10, branch_length=300, rand_multiplier=2, seed=37, sigma=4):

    return phate.tree.gen_dla(n_dim=n_dim, n_branch=n_branch, branch_length=branch_length,
                              rand_multiplier=rand_multiplier, seed=seed, sigma=sigma)


def make_missing_middle(data, n_missing, idxs=None):
    """
    Take a dataset and create two datasets, one with missing (NaN)
    values in the interquartile range, another with all NaNs replaced
    with the mean.

    @param data the data matrix
    @param n_missing an array with numbers of observations with missing values in each column
    @param idxs the column indices to do this process in
    @return two data matrices
    """

    d = data.shape[1]
    n = data.shape[0]

    if idxs is None:
        idxs = np.arange(d)
    elif len(idxs) > d:
        print("Error: Too many values selected")
        return

    new_missing, new_imputed = np.copy(data), np.copy(data)

    for idx, num in zip(idxs, n_missing):
        col = new_missing[:, idx]
        q1 = np.quantile(col, 0.25)
        q3 = np.quantile(col, 0.75)

        mask = (col >= q1) & (col <= q3)
        where_iqr = np.nonzero(mask)[0]
        to_del = np.random.choice(where_iqr, size=num, replace=False)

        new_missing[to_del, idx] = np.nan
        new_imputed[to_del, idx] = np.nanmean(new_missing[:, idx])

    new_missing = preprocessing.scale(new_missing, with_mean=True, with_std=True)
    new_imputed = preprocessing.scale(new_imputed, with_mean=True, with_std=True)

    return new_missing, new_imputed


def make_missing_random(data, n_missing, idxs=None):
    """
    Take a dataset and create two datasets, one with missing (NaN)
    values randomly, another with all NaNs replaced with the mean.

    @param data the data matrix
    @param n_missing an array with numbers of observations with missing values in each column
    @param idxs the column indices to do this process in
    @return two data matrices
    """

    d = data.shape[1]
    n = data.shape[0]

    if idxs is None:
        idxs = np.arange(d)
    elif len(idxs) > d:
        print("Error: Too many values selected")
        return

    new_missing, new_imputed = np.copy(data), np.copy(data)

    for idx, num in zip(idxs, n_missing):
        to_del = np.random.choice(np.arange(n), size=num, replace=False)

        new_missing[to_del, idx] = np.nan
        new_imputed[to_del, idx] = np.nanmean(new_missing[:, idx])

    new_missing = preprocessing.scale(new_missing, with_mean=True, with_std=True)
    new_imputed = preprocessing.scale(new_imputed, with_mean=True, with_std=True)

    return new_missing, new_imputed


def make_embeddings(data, labels, path=None):
    """
    Make PCA, tSNE, and PHATE embeddings of the dataset and save them.

    @param data the data to apply it to
    @param labels the labels used for coloring
    @param path the path to save embeddings and plots in, CWD by default
    """
    
    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    if not os.path.isdir(f"{path}/non_mural"):
        os.mkdir(f"{path}/non_mural")

    dims = data.shape[1]

    # PCA reduction 
    data_pca = scprep.reduce.pca(data, dims)
    scprep.plot.scatter2d(data_pca[:, 0:2], c = labels,
                          legend_anchor=(1,1),
                          label_prefix='PCA', ticks=None,
                          title='PCA',
                          figsize=(7,5),
                          filename=f"{path}/non_mural/pca.png")
    np.save(f"{path}/non_mural/pca.npy", data_pca[:, 0:2])

    # tSNE
    data_tsne = TSNE(n_components=2).fit_transform(data)
    scprep.plot.scatter2d(data_tsne, c = labels,
                          legend_anchor=(1,1),
                          label_prefix='tSNE', ticks=None,
                          title='tSNE',
                          figsize=(7,5),
                          filename=f"{path}/non_mural/tsne.png")
    np.save(f"{path}/non_mural/tsne.npy", data_tsne)

    # PHATE
    phate_op = phate.PHATE()
    phate_orig = phate_op.fit_transform(data)
    scprep.plot.scatter2d(phate_orig, c = labels,
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename=f"{path}/non_mural/phate.png")
    np.save(f"{path}/non_mural/phate.npy", phate_orig)


def test_forest(forest, data, labels, path=None, geometric=False):
    """
    Perform tests for a MURAL forest and saves embeddings and plots.

    @param forest a fitted UnsupervisedForest to test
    @param data the data to apply it to
    @param labels the labels used for coloring
    @param path the path to save embeddings and plots in, CWD by default
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    num_trees = forest.n_estimators
    depth = forest.depth

    if not os.path.isdir(f"{path}/{num_trees}trees{depth}depth"):
        os.mkdir(f"{path}/{num_trees}trees{depth}depth")

    # Run binary affinities
    forest_fitted = forest.apply(data)
    b_forest = binary_affinity(forest_fitted)

    # Plot binary affinities with PHATE by passing in affinity matrix
    phate_b = phate.PHATE(knn_dist="precomputed_affinity")
    phate_fit_b = phate_b.fit_transform(b_forest)
    np.save(f"{path}/{num_trees}trees{depth}depth/binary_mural", phate_fit_b)

    scprep.plot.scatter2d(phate_fit_b, c = labels, 
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename=f"{path}/{num_trees}trees{depth}depth/binary")

    # For exponential affinities
    weighted_D_list = [adjacency_to_distances(A, L, geometric=geometric, weighted=True) for A, L in zip(forest.adjacency(), forest.leaves())]
    weighted_avg_distance = get_average_distance(weighted_D_list, forest_fitted)

    # Plot exponential affinities with PHATE by passing in distance matrix
    phate_w_e = phate.PHATE(knn_dist="precomputed_distance")
    phate_fit_w_e = phate_w_e.fit_transform(weighted_avg_distance)
    np.save(f"{path}/{num_trees}trees{depth}depth/exponential_weighted_mural", phate_fit_w_e)
    scprep.plot.scatter2d(phate_fit_w_e, c = labels, 
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename=f"{path}/{num_trees}trees{depth}depth/exponential_weighted")

    # Try unweighted edges
    unweighted_D_list = [adjacency_to_distances(A, L, geometric=geometric, weighted=False) for A, L in zip(forest.adjacency(), forest.leaves())]
    unweighted_avg_distance = get_average_distance(unweighted_D_list, forest_fitted)

    # Plot exponential affinities with PHATE
    phate_uw_e = phate.PHATE(knn_dist="precomputed_distance")
    phate_fit_uw_e = phate_uw_e.fit_transform(unweighted_avg_distance)
    np.save(f"{path}/{num_trees}trees{depth}depth/exponential_unweighted_mural", phate_fit_uw_e)
    scprep.plot.scatter2d(phate_fit_uw_e, c = labels, 
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename=f"{path}/{num_trees}trees{depth}depth/exponential_unweighted")

    
def demap_reference(data, path=None):
    """
    Run DEMaP on dataset and reference embeddings and save the results.

    @param data the data matrix
    @param path a path to directory containing a non_mural directory
    with the embeddings
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    if not os.path.isdir(f"{path}/non_mural"):
        os.mkdir(f"{path}/non_mural")

    data_pca = np.load(f"{path}/non_mural/pca.npy")
    data_tsne = np.load(f"{path}/non_mural/tsne.npy")
    data_phate = np.load(f"{path}/non_mural/phate.npy")

    demap_pca = demap.DEMaP(data, data_pca)
    demap_tsne = demap.DEMaP(data, data_tsne)
    demap_phate = demap.DEMaP(data, data_phate)

    f = open(f"{path}/demap_reference.txt", "w")
    f.write(f"DEMaP for PCA is {demap_pca}\n")
    f.write(f"DEMaP for tSNE is {demap_tsne}\n")
    f.write(f"DEMaP for PHATE is {demap_phate}\n")
    f.close()


def demap_mural(data, path=None, trees=default_trees, depths=default_depths):
    """
    Run DEMaP on MURAL embeddings and save the results.

    @param gt the ground truth
    @param path the path to a directory with the MURAL embedding directories
    @param trees an array of numbers of trees
    @param depths an array of depth parameters
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    f = open(f"{path}/demap_mural.txt", "w")

    for t in trees:
        for d in depths:
            exponential_weighted_mural = np.load(f"{path}/{t}trees{d}depth/exponential_weighted_mural.npy")
            exponential_unweighted_mural = np.load(f"{path}/{t}trees{d}depth/exponential_unweighted_mural.npy")
            binary_mural = np.load(f"{path}/{t}trees{d}depth/binary_mural.npy")

            demap_mural_w_e = demap.DEMaP(data, exponential_weighted_mural)
            demap_mural_uw_e = demap.DEMaP(data, exponential_unweighted_mural)
            demap_mural_b = demap.DEMaP(data, binary_mural)

            f.write(f"For {t} trees, {d} depth:\n")
            f.write(f"DEMaP for MURAL (w, e) is {demap_mural_w_e}\n")
            f.write(f"DEMaP for MURAL (uw, e) is {demap_mural_uw_e}\n")
            f.write(f"DEMaP for MURAL (b) is {demap_mural_b}\n")
            f.write("\n")

    f.close()


def train_forests(data, labels, sampled_features, batch_size, min_leaf_size=2,
                  decay=0.5, t_list=default_trees, d_list=default_depths, path=None,
                  missing_profile=1, weighted=True, geometric=False, optimize="max",
                  use_missing=False, entropy="one", imputed=None, avoid=None, layers=1,
                  quad=False):
    """
    Train MURAL forests and save them and their embeddings.

    @param data the data matrix to train MURAL forests on
    @param labels the labels used for coloring plots
    @param sampled_features the number of features for each node to randomly look at
    @param batch_size the number of observations to subsample to
    @param min_leaf size the minimum number of observations needed to split
    @param decay a parameter that controls how quickly the weight in the missing values node decays
    @param t_list an array of numbers of trees
    @param d_list an array of depth parameters
    @param path the path to a directory to save MURAL embeddings in
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    f = open(f"{path}/times.txt", "w")

    for t in t_list:
        for d in d_list:
            if not os.path.isdir(f"{path}/{t}trees{d}depth"):
                os.mkdir(f"{path}/{t}trees{d}depth")

            forest = UnsupervisedForest(data, t, sampled_features, batch_size, depth=d, 
                                        min_leaf_size=min_leaf_size, decay=decay, imputed=imputed,
                                        missing_profile=missing_profile, weighted=weighted, optimize=optimize,
                                        use_missing=use_missing, entropy=entropy, avoid=avoid, layers=layers,
                                        quad=quad)
            forest.to_pickle(f"{path}/{t}trees{d}depth/forest.pkl")
            f.write(f"Training time for {t} trees, {d} depth: {forest.time_used:0.4f} seconds\n")

            test_forest(forest, data, labels, path, geometric=geometric)

    f.close()


def accuracy(A, A_true, n_neighbors, n_distributions):
    """
    Determine accuracy of kNN graph generated by embedding and ground truth 
    Save the results.
    @param A the kNN graph
    @param A_true the ground truth kNN graph
    @param n_neighbors number of neighbors for kNN graph
    @param n_distributions number of nodes in graph, generally dimension of array
    """

    accuracy = np.sum((A_true + A) == 2) / (n_neighbors * n_distributions)
    return accuracy


def knn_graph(data, n_neighbors):
    """
    Create kNN graph on dataset and save the results.
    @param data the data matrix
    @param n_neighbors number of neighbors for kNN graph
    """

    A = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
    data_knn = A.toarray()
    return data_knn


def knn_reference(gt_knn, n_neighbors, path=None):
    """
    Create kNN graph on dataset and reference embeddings and save the results.
    @param gt_knn the ground truth kNN graph
    @param n_neighbors number of neighbors for kNN graph
    @param path a path to directory containing a non_mural directory
    with the embeddings
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    if not os.path.isdir(f"{path}/non_mural"):
        os.mkdir(f"{path}/non_mural")
    
    data_pca = np.load(f"{path}/non_mural/pca.npy")
    data_tsne = np.load(f"{path}/non_mural/tsne.npy")
    data_phate = np.load(f"{path}/non_mural/phate.npy")
    
    pca_knn = knn_graph(data_pca, n_neighbors)
    tsne_knn = knn_graph(data_tsne, n_neighbors)
    phate_knn = knn_graph(data_phate, n_neighbors)

    acc_pca = accuracy(pca_knn, gt_knn, n_neighbors, pca_knn.shape[1])
    acc_tsne = accuracy(tsne_knn, gt_knn, n_neighbors, tsne_knn.shape[1])
    acc_phate = accuracy(phate_knn, gt_knn, n_neighbors, phate_knn.shape[1])

    f = open(f"{path}/knn_reference_{n_neighbors}.txt", "w")
    f.write(f"Accuracy via kNN for PCA with {n_neighbors} neighbors is {acc_pca}\n")
    f.write(f"Accuracy via kNN for tSNE with {n_neighbors} neighbors is {acc_tsne}\n")
    f.write(f"Accuracy via kNN for PHATE with {n_neighbors} neighbors is {acc_phate}\n")
    f.close()


def knn_mural(gt_knn, n_neighbors, path=None, trees=default_trees, depths=default_depths):
    """
    Create kNN graph on dataset and reference embeddings and save the results.
    @param data the data matrix
    @param gt_knn the ground truth kNN graph
    @param n_neighbors number of neighbors for kNN graph
    @param path the path to a directory with the MURAL embedding directories
    @param trees an array of numbers of trees
    @param depths an array of depth parameters
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    f = open(f"{path}/knn_mural_{n_neighbors}.txt", "w")

    for t in trees:
        for d in depths:
            exponential_weighted_mural = np.load(f"{path}/{t}trees{d}depth/exponential_weighted_mural.npy")
            exponential_unweighted_mural = np.load(f"{path}/{t}trees{d}depth/exponential_unweighted_mural.npy")
            binary_mural = np.load(f"{path}/{t}trees{d}depth/binary_mural.npy")

            knn_mural_w_e = knn_graph(exponential_weighted_mural, n_neighbors)
            knn_mural_uw_e = knn_graph(exponential_unweighted_mural, n_neighbors)
            knn_mural_b = knn_graph(binary_mural, n_neighbors)	    

            acc_mural_w_e = accuracy(knn_mural_w_e, gt_knn, n_neighbors, knn_mural_w_e.shape[1])
            acc_mural_uw_e = accuracy(knn_mural_uw_e, gt_knn, n_neighbors, knn_mural_uw_e.shape[1])
            acc_mural_b = accuracy(knn_mural_b, gt_knn, n_neighbors, knn_mural_b.shape[1])
  
            f.write(f"For {t} trees, {d} depth:\n")
            f.write(f"Accuracy via kNN for MURAL (w, e) with {n_neighbors} neighbors is {acc_mural_w_e}\n")
            f.write(f"Accuracy via kNN for MURAL (w, e) with {n_neighbors} neighbors is {acc_mural_uw_e}\n")
            f.write(f"Accuracy via kNN for MURAL (b) with {n_neighbors} neighbors is {acc_mural_b}\n")
            f.write("\n")

    f.close()
