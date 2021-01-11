# Code for testing MURAL
import base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scprep
import phate
from sklearn.manifold import TSNE
from sklearn import preprocessing
from scipy.stats import special_ortho_group
from sklearn.datasets import make_swiss_roll
import demap
import os

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
    x = np.dot(x, special_ortho_group.rvs(dims)[:3])

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

    new_missing, new_imputed = data[:,:], data[:,:]

    for idx, num in zip(idxs, n_missing):
        col = new_missing[:, idx]
        avg = np.mean(col)
        q1 = np.quantile(col, 0.25)
        q3 = np.quantile(col, 0.75)

        mask = (col >= q1) & (col <= q3)
        where_iqr = np.nonzero(mask)[0]
        to_del = np.random.choice(where_iqr, size=num, replace=False)

        new_missing[to_del, idx] = None
        new_imputed[to_del, idx] = avg

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


def test_forest(forest, data, labels, path=None):
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
    b_forest = base.binary_affinity(forest_fitted)

    # Plot binary affinities with PHATE by passing in affinity matrix
    phate_b = phate.PHATE(knn_dist="precomputed_distance")
    phate_fit_b = phate_b.fit_transform(b_forest)
    np.save(f"{path}/{num_trees}trees{depth}depth/binary_mural", phate_fit_b)

    scprep.plot.scatter2d(phate_fit_b, c = labels, 
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename=f"{path}/{num_trees}trees{depth}depth/binary")

    # For exponential affinities
    D_list = [base.adjacency_to_distances(A) for A in forest.adjacency()]
    avg_distance = base.get_average_distance(D_list, forest_fitted)

    # Plot exponential affinities with PHATE by passing in distance matrix
    phate_e = phate.PHATE(knn_dist="precomputed_distance")
    phate_fit_e = phate_e.fit_transform(avg_distance)
    np.save(f"{path}/{num_trees}trees{depth}depth/exponential_mural", phate_fit_e)
    scprep.plot.scatter2d(phate_fit_e, c = labels, 
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename=f"{path}/{num_trees}trees{depth}depth/exponential")

    
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
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    f = open(f"{path}/demap_mural.txt", "w")

    for t in trees:
        for d in depths:
            exponential_mural = np.load(f"{path}/{t}trees{d}depth/exponential_mural.npy")
            binary_mural = np.load(f"{path}/{t}trees{d}depth/binary_mural.npy")

            demap_mural_e = demap.DEMaP(data, exponential_mural)
            demap_mural_b = demap.DEMaP(data, binary_mural)

            f.write(f"For {t} trees, {d} depth:\n")
            f.write(f"DEMaP for MURAL (e) is {demap_mural_e}\n")
            f.write(f"DEMaP for MURAL (b) is {demap_mural_b}\n")
            f.write("\n")

    f.close()


def train_forests(data, labels, sampled_features, batch_size, min_leaf_size=2, 
                  decay=0.5, t_list=default_trees, d_list=default_depths, path=None):
    """
    Train MURAL forests and save them and their embeddings.
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

            forest = base.UnsupervisedForest(data, t, sampled_features, batch_size, d, min_leaf_size, decay)
            forest.to_pickle(f"{path}/{t}trees{d}depth/forest.pkl")
            f.write(f"Training time for {t} trees, {d} depth: {forest.time_used:0.4f} seconds\n")

            test_forest(forest, data, labels, path)

    demap_mural(data, path, t_list, d_list)

    f.close()
