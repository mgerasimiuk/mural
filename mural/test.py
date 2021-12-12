# Code for testing MURAL
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scprep
import phate
from sklearn.manifold import TSNE
from sklearn import preprocessing
from scipy.stats import special_ortho_group, spearmanr
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
    Makes plots of data distributions.

    Arguments:
    df -- a pandas DataFrame
    vars_list -- a list of variables to visualize
    """
    df[vars_list].hist(figsize=(16, 20), bins=100, xlabelsize=8, ylabelsize=8)


def make_embedded_swiss_roll(dims=3):
    """
    Makes a dataset of the Swiss roll manifold embedded in multiple dimensions.

    Arguments:
    dims -- the number of dimensions to embed in
    
    Return:
    the data matrix for the embedded Swiss roll
    """
    assert dims >= 3

    # Generate Swiss Roll
    x, labels = make_swiss_roll(n_samples=3000, random_state=42)
    if dims > 3:
        x = np.dot(x, special_ortho_group.rvs(dims)[:3])

    # Standardize with mean and standard deviation
    standardized_X = preprocessing.scale(x, with_mean=True, with_std=True)

    return standardized_X, labels


def make_embeddings(data, labels, path=None):
    """
    Makes PCA, tSNE, and PHATE embeddings of the dataset and saves them.

    Arguments:
    data -- the data to apply it to
    labels -- the labels used for coloring
    path -- the path to save embeddings and plots in, CWD by default
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
    Performs tests for a MURAL forest and saves embeddings and plots.

    Arguments:
    forest -- a fitted UnsupervisedForest to test
    data -- the data to apply it to
    labels -- the labels used for coloring
    path -- the path to save embeddings and plots in, CWD by default
    geometric -- ignore
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
    np.save(f"{path}/{num_trees}trees{depth}depth/binary_affinity.npy", b_forest)

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
    np.save(f"{path}/{num_trees}trees{depth}depth/weighted_distance.npy", weighted_avg_distance)

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
    np.save(f"{path}/{num_trees}trees{depth}depth/unweighted_distance.npy", unweighted_avg_distance)

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


def save_forest(forest, data, path=None, geometric=False):
    """
    Saves embeddings and plots for a trained MURAL forest.

    Arguments:
    forest -- a fitted UnsupervisedForest to test
    data -- the data to apply it to
    labels -- the labels used for coloring
    path -- the path to save embeddings and plots in, CWD by default
    geometric -- ignore
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
    np.save(f"{path}/{num_trees}trees{depth}depth/binary_affinity.npy", b_forest)

    # For exponential affinities
    weighted_D_list = [adjacency_to_distances(A, L, geometric=geometric, weighted=True) for A, L in zip(forest.adjacency(), forest.leaves())]
    weighted_avg_distance = get_average_distance(weighted_D_list, forest_fitted)
    np.save(f"{path}/{num_trees}trees{depth}depth/weighted_distance.npy", weighted_avg_distance)

    # Try unweighted edges
    unweighted_D_list = [adjacency_to_distances(A, L, geometric=geometric, weighted=False) for A, L in zip(forest.adjacency(), forest.leaves())]
    unweighted_avg_distance = get_average_distance(unweighted_D_list, forest_fitted)
    np.save(f"{path}/{num_trees}trees{depth}depth/unweighted_distance.npy", unweighted_avg_distance)

    
def demap_reference(data, path=None):
    """
    Runs DEMaP on dataset and reference embeddings and saves the results.

    Arguments:
    data -- the data matrix
    path -- a path to directory containing a non_mural directory with the embeddings
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

    data -- the ground truth
    path -- the path to a directory with the MURAL embedding directories
    trees -- a list of numbers of trees
    depths -- a list of depth parameters
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    f = open(f"{path}/demap_mural.txt", "a")

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
                  quad=False, b_ind=None, m_ind=None, avoid_binary=False, rand_entropy=None):
    """
    Trains MURAL forests and saves them and their embeddings.

    Arguments:
    data -- the data matrix to train MURAL forests on
    labels -- the labels used for coloring plots
    sampled_features -- the number of features for each node to randomly look at
    batch_size -- the number of observations to subsample to
    min_leaf_size -- the minimum number of observations needed to split
    decay -- ignore
    t_list -- a list of numbers of trees
    d_list -- a list of depth parameters
    path -- the path to a directory to save MURAL embeddings in (default CWD)
    missing_profile -- ignore
    weighted -- ignore
    geometric -- ignore
    optimize -- ignore
    use_missing -- ignore
    entropy -- the type of entropy to use (one|two|three|many|full)
    imputed -- ignore
    avoid -- specify "hard" to avoid choosing variables with missing values in some levels
    layers -- number of tree depth levels to apply chosen avoid policy in
    quad -- ignore
    b_ind -- list with indices of binary variables
    m_ind -- list with indices of variables with missing values
    avoid_binary -- whether to apply avoid policy to binary variables as well
    rand_entropy -- entropy to be given to SeedSequence
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    f = open(f"{path}/times.txt", "a")

    for t in t_list:
        for d in d_list:
            if not os.path.isdir(f"{path}/{t}trees{d}depth"):
                os.mkdir(f"{path}/{t}trees{d}depth")

            forest = UnsupervisedForest(data, t, sampled_features, batch_size, depth=d, 
                                        min_leaf_size=min_leaf_size, decay=decay, imputed=imputed,
                                        missing_profile=missing_profile, weighted=weighted, optimize=optimize,
                                        use_missing=use_missing, entropy=entropy, avoid=avoid, layers=layers,
                                        quad=quad, b_ind=b_ind, m_ind=m_ind, avoid_binary=avoid_binary, rand_entropy=rand_entropy)
            forest.to_pickle(f"{path}/{t}trees{d}depth/forest.pkl")
            f.write(f"Training time for {t} trees, {d} depth: {forest.time_used:0.4f} seconds\n")

            test_forest(forest, data, labels, path, geometric=geometric)

    f.close()


def accuracy(A, A_true, n_neighbors, n_distributions):
    """
    Determines accuracy of kNN graph generated by embedding and ground truth.
    
    Arguments:
    A -- the kNN graph
    A_true -- the ground truth kNN graph
    n_neighbors -- number of neighbors for kNN graph
    n_distributions -- number of nodes in graph, generally dimension of array
    
    Return:
    accuracy
    """

    accuracy = np.sum((A_true + A) == 2) / (n_neighbors * n_distributions)
    return accuracy


def knn_graph(data, n_neighbors):
    """
    Creates kNN graph on dataset and returns the results.
    
    Arguments:
    data -- the data matrix
    n_neighbors -- number of neighbors for kNN graph
    
    Return:
    the kNN graph
    """

    A = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
    data_knn = A.toarray()
    return data_knn


def knn_reference(gt_knn, n_neighbors, path=None):
    """
    Creates kNN graph on dataset and reference embeddings and saves the results.
    
    Arguments:
    gt_knn -- the ground truth kNN graph
    n_neighbors -- number of neighbors for kNN graph
    path -- a path to directory containing a non_mural directory
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
    Creates kNN graph on dataset and reference embeddings and saves the results.
    
    Arguments:
    data -- the data matrix
    gt_knn -- the ground truth kNN graph
    n_neighbors -- number of neighbors for kNN graph
    path -- the path to a directory with the MURAL embedding directories
    trees -- an array of numbers of trees
    depths -- an array of depth parameters
    """

    if path is None:
        path = os.getcwd()
    elif not os.path.isdir(path):
        print("Error: Path does not exist")
        return

    f = open(f"{path}/knn_mural_{n_neighbors}.txt", "a")
    g = open(f"{path}/knn_mural_{n_neighbors}.csv", "a")
    g.write(f"n_neighbors;num_trees;depth;affinity_type;mural_acc\n")

    for t in trees:
        for d in depths:
            b_aff = np.load(f"{path}/{t}trees{d}depth/binary_affinity.npy")
            w_dist = np.load(f"{path}/{t}trees{d}depth/weighted_distance.npy")
            uw_dist = np.load(f"{path}/{t}trees{d}depth/unweighted_distance.npy")

            knn_mural_b = knn_graph(b_aff, n_neighbors)
            knn_mural_we = knn_graph(w_dist, n_neighbors)
            knn_mural_uwe = knn_graph(uw_dist, n_neighbors)   

            acc_mural_b = accuracy(knn_mural_b, gt_knn, n_neighbors, knn_mural_b.shape[1])
            acc_mural_we = accuracy(knn_mural_we, gt_knn, n_neighbors, knn_mural_we.shape[1])
            acc_mural_uwe = accuracy(knn_mural_uwe, gt_knn, n_neighbors, knn_mural_uwe.shape[1])
            
            f.write(f"For {t} trees, {d} depth:\n")
            f.write(f"Accuracy via kNN for MURAL (b) with {n_neighbors} neighbors is {acc_mural_b}\n")
            f.write(f"Accuracy via kNN for MURAL (w, e) with {n_neighbors} neighbors is {acc_mural_we}\n")
            f.write(f"Accuracy via kNN for MURAL (uw, e) with {n_neighbors} neighbors is {acc_mural_uwe}\n")
            f.write("\n")

            g.write(f"{n_neighbors};{t};{d};b;{acc_mural_b}\n")
            g.write(f"{n_neighbors};{t};{d};we;{acc_mural_we}\n")
            g.write(f"{n_neighbors};{t};{d};uwe;{acc_mural_uwe}\n")

    f.close()
    g.close()


def dists_to_ranks(dists):
    order = dists.argsort(axis=0)
    ranks = order.argsort(axis=0)
    return ranks
    
    
def corrs(d1, d2):
    spearman_corrs = []
    dr1 = dists_to_ranks(d1)
    dr2 = dists_to_ranks(d2)
    for i in range(len(d1)):
        #print(i)
        correlation, pvale = spearmanr(
            dr1[i], dr2[i]
        )
        spearman_corrs.append(correlation)
    spearman_corrs = np.array(spearman_corrs)
    return np.mean(spearman_corrs)
#k_neighbors from distance matrix.


def dists_to_knn(dists, k):
    n = dists.shape[0]
    return np.eye(n)[np.argsort(dists)[:, :k+1]].sum(axis=1) - np.eye(n)
    
    
def p_at_n(dists, gt_dists, k=10):
    d_a = np.argsort(dists)[:, :k+1]
    d_g = np.argsort(gt_dists)[:, :k+1]
    n = dists.shape[0]
    sums = 0
    for l_a, l_g in zip(d_a, d_g):
        #print(len(l_a), len(l_g), len(np.intersect1d(l_a, l_g)))
        sums += (len(np.intersect1d(l_a, l_g)) - 1) / min(k, n - 1) # Count how many are the same
    accuracy = sums / n
    return accuracy

    
def distortion(dists, gt_dists):
    # Normalize the distances such that r * dists is always <= gt_dists
    eps = 1e-8
    r = 1 / (np.nanmax(dists / (gt_dists + eps)))
    norm_dists = dists * r
    c = np.nanmax(gt_dists / (norm_dists + eps))
    #print(r,c)
    return c
    
    
def dist_to_line(name, dists, gt_dists):
    spearman = corrs(dists, gt_dists)
    p_at_5 = p_at_n(dists, gt_dists, k=5)
    p_at_10 = p_at_n(dists, gt_dists)
    p_at_100 = p_at_n(dists, gt_dists, k=100)
    p_at_500 = p_at_n(dists, gt_dists, k=500)
    dist_coeff = distortion(dists, gt_dists)
    return pd.DataFrame([name, spearman, p_at_5, p_at_10, p_at_100, p_at_500, dist_coeff], 
                        index = ["name", "spearman", "P@5", "P@10", "P@100", "P@500", "Distortion"]).T


