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
    elif not os.path.isdir(f"{path}"):
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
    np.save("f{path}/{num_trees}trees{depth}depth/binary_mural", phate_fit_b)

    scprep.plot.scatter2d(phate_fit_b, c = labels, 
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename="f{path}/{num_trees}trees{depth}depth/binary")

    # For exponential affinities
    D_list = [base.adjacency_to_distances(A) for A in forest.adjacency()]
    avg_distance = base.get_average_distance(D_list, forest_fitted)

    # Plot exponential affinities with PHATE by passing in distance matrix
    phate_e = phate.PHATE(knn_dist="precomputed_distance")
    phate_fit_e = phate_e.fit_transform(avg_distance)
    np.save("f{path}/{num_trees}trees{depth}depth/exponential_mural", phate_fit_e)
    scprep.plot.scatter2d(phate_fit_e, c = labels, 
                          legend_anchor=(1,1),
                          label_prefix='PHATE', ticks=None,
                          title='PHATE',
                          figsize=(7,5),
                          filename="f{path}/{num_trees}trees{depth}depth/exponential")
