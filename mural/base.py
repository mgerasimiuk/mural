# MURAL
# Based on the structure proposed by Vaibhav Kumar 
# (https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249)

import time
import numpy as np
from collections import deque
import pickle
import json
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from numpy.core.fromnumeric import size
from numpy.lib.polynomial import roots
from sklearn import impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from _entropy import *
from _affinity import *
from _utils import *

EPSILON = np.finfo(float).eps
N_JOBS = min(multiprocessing.cpu_count(), 8)


class UnsupervisedForest():
    """
    Realization of the random forest classifier for the unsupervised setting with samples involving missingness.
    """
    def __init__(self, X, n_estimators, n_sampled_features, batch_size, imputed=None, depth=4, min_leaf_size=2, 
                decay=None, missing_profile=1, weighted=True, optimize="max", entropy="one", use_missing=True,
                avoid=None, layers=1, quad=False, m_ind=None, b_ind=None, avoid_binary=False, rand_entropy=None):
        """
        Creates and fits a MURAL forest.
        
        Arguments:
        X -- the data matrix to train MURAL forests on
        n_estimators -- number of trees
        sampled_features -- the number of features for each node to randomly look at
        batch_size -- the number of observations to subsample to
        imputed -- ignore
        depth -- the maximum depth of each tree
        min_leaf_size -- the minimum number of observations needed to split
        decay -- ignore
        missing_profile -- ignore
        weighted -- ignore
        optimize -- ignore
        entropy -- the type of entropy to use (one|two|three|many|full)
        use_missing -- ignore
        avoid -- specify "hard" to avoid choosing variables with missing values in some levels
        layers -- number of tree depth levels to apply chosen avoid policy in
        quad -- ignore
        m_ind -- list with indices of variables with missing values
        b_ind -- list with indices of binary variables
        avoid_binary -- whether to apply avoid policy to binary variables as well
        rand_entropy -- entropy to be given to SeedSequence
        """

        # Measure time to evaluate model efficiency
        tic = time.perf_counter()
        if batch_size is None:
            # Use all the observations each time
            self.batch_size = X.shape[0]
        else:
            self.batch_size = batch_size
        
        assert (self.batch_size <= X.shape[0] and min_leaf_size <= X.shape[0])

        if n_sampled_features == "log":
            self.n_sampled_features = int(np.log2(X.shape[1]))
        elif n_sampled_features == "sqrt":
            self.n_sampled_features = int(np.sqrt(X.shape[1]))
        else:
            assert (n_sampled_features <= X.shape[1])
            self.n_sampled_features = n_sampled_features

        if type(missing_profile) == int:
            if missing_profile == 1:
                self.missing_profile = np.ones(shape=X.shape[1])
            elif missing_profile == 0:
                self.missing_profile = np.zeros(shape=X.shape[1])
            else:
                raise ValueError
        else:
            assert (len(missing_profile) == X.shape[1])
            assert (np.all(missing_profile >= 0) and np.all(missing_profile <= 1))
            self.missing_profile = missing_profile

        self.X = X
        self.imputed = imputed
        self.n_estimators = n_estimators
        self.depth = depth
        self.min_leaf_size = min_leaf_size
        self.avoid = avoid
        self.avoid_binary = avoid_binary
        self.layers = layers
        self.quad = quad
        
        if m_ind is None:
            self.m_ind = np.any(np.isnan(X), axis=0)
        else:
            self.m_ind = m_ind
        
        if b_ind is None:
            mask = np.any(~np.isnan(X) & (X != 0), axis=0) & np.any(~np.isnan(X) & (X != 1), axis=0)
            self.b_ind = np.ones(shape=X.shape[1])
            self.b_ind[mask] = 0
        else:
            self.b_ind = b_ind

        # Experiment with unweighted (all 1) edges
        self.weighted = weighted
        # Experiment with different objective
        assert (optimize == "max" or optimize == "min")
        self.optimize = optimize
        self.use_missing = use_missing

        self.entropy = entropy

        if decay is None:
            self.decay = 0.5
        else:
            self.decay = decay

        self.rng = np.random.default_rng()

        if rand_entropy is None:
            rand_entropy = 12345
        ss = np.random.SeedSequence(rand_entropy)
        child_seeds = ss.spawn(n_estimators)
        streams = [np.random.default_rng(s) for s in child_seeds]

        imputers = np.arange(n_estimators) # ignore
        self.trees = Parallel(n_jobs=N_JOBS)(delayed(self.create_tree)(stream, imputers[i], i) for i, stream in enumerate(streams))
        
        toc = time.perf_counter()
        self.time_used = toc - tic
        print(f"Completed the Unsupervised RF in {self.time_used:0.4f} seconds")
        
    def create_tree(self, rng, imputer, root_index=0):
        """
        Creates and fits a decision tree to be used by the forest.

        Arguments:
        rng -- a random number generator
        imputer -- ignore
        root_index -- nothing, used to parallelize
        
        Return:
        an UnsupervisedTree object
        """

        replace = False
        if self.avoid is None:
            prob = None
        elif self.avoid == "soft":
            prob = np.count_nonzero(~np.isnan(self.X), axis=0) # Probability that each variable is not missing
            p_sum = np.sum(prob)
            if p_sum == 0:
                prob = None
            else:
                prob = prob / p_sum
        elif self.avoid == "hard" or self.avoid == "hard2" or self.avoid == "mix" or self.avoid == "mix2":
            prob = np.ones(shape=self.X.shape[1])
            mask = np.any(np.isnan(self.X), axis=0)
            if self.avoid_binary:
                mask = mask | (self.b_ind == 1)
            prob[mask] = 0
            p_sum = np.sum(prob)
            if p_sum == 0:
                prob = None
            else:
                prob = prob / p_sum
        else:
            prob = None

        chosen_features = np.sort(rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=replace))
        chosen_inputs = np.sort(rng.choice(self.X.shape[0], size=self.batch_size, replace=replace))

        if self.imputed is None:
            imputed = self.X # This is to be removed later
        else:
            imputed = self.imputed

        return UnsupervisedTree(self.X, imputed, self.n_sampled_features, chosen_inputs, chosen_features,
                                depth=self.depth, min_leaf_size=self.min_leaf_size, weight=2 ** (self.depth - 1),
                                m_ind=self.m_ind, b_ind=self.b_ind, decay=self.decay, entropy=self.entropy, optimize=self.optimize,
                                use_missing=self.use_missing, rng=rng, imputer=imputer, forest=self,
                                root_index=root_index, avoid=self.avoid, layers=self.layers, quad=self.quad, avoid_binary=self.avoid_binary)
    
    def apply(self, x):
        """
        Applies the fitted model to data.
        
        Arguments:
        x the data to apply the tree to
        
        Return:
        leaf classifications of each datapoint
        """

        result = Parallel(n_jobs=N_JOBS)(delayed(tree.apply)(x) for tree in self.trees)

        return result
    
    def adjacency(self):
        """
        Gets adjacency lists from each tree in a fitted model.
        
        Return:
        adjacency lists representing all trees
        """

        result = Parallel(n_jobs=N_JOBS)(delayed(tree.adjacency)(i) for i, tree in enumerate(self.trees))

        return result

    def leaves(self):
        """
        Gets lists of leaves for each tree in a fitted model.
        
        Return:
        lists of indices (in the respective adjacency lists)
        of terminal nodes
        """

        result = Parallel(n_jobs=N_JOBS)(delayed(tree.leaves)(i) for i, tree in enumerate(self.trees))

        return result

    def draw(self, n=None):
        """
        A simple way to visualize n trees from the forest.
        """

        N = len(self.trees)
        if n is None:
            n = N
        elif n > N:
            print(f"The most you can print are {N} trees")
            return
        
        for i in range(n):
            plt.figure(i)
            self.trees[i].draw()

    def wasserstein(self, p, q):
        """
        Computes the tree-sliced Wasserstein metric for the given cohorts.

        Arguments:
        p -- a cohort
        q -- another cohort
        
        Return:
        an estimate of the Wasserstein distance between them, array of averaged importances
        """
        W_list = np.array(Parallel(n_jobs=N_JOBS)(delayed(tree.wasserstein)(p, q) for tree in self.trees))
        W_vals = W_list[:, 0]
        W_imps = np.array(W_list[:, 1])
        return sum(W_vals) / len(self.trees), np.mean(W_imps, axis=0)

    def to_pickle(self, path):
        """
        Saves the object to a pickle with the given path.
        
        Arguments:
        path -- a path
        """

        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def to_json(self, path):
        """
        Saves the object to a json file with the given path.
        
        Arguments:
        path -- a path
        """

        f = open(path, "w")
        json.dump(self, f)
        f.close()


class UnsupervisedTree():
    """
    Decision tree class for use in unsupervised learning of data involving missingness.
    """
    def __init__(self, X, imputed, n_sampled_features, chosen_inputs, chosen_features, depth, min_leaf_size, 
                 weight, m_ind=None, b_ind=None, decay=0.5, entropy="one", optimize="max", use_missing=True, avoid=None, layers=1,
                 quad=False, override=None, rng=None, imputer=None, root=None, parent=None, forest=None, 
                 root_index=0, mode=1, avoid_binary=False):
        """
        Creates and fits a MURAL decision tree.
        
        Arguments:
        X -- the data matrix to train the MURAL tree on
        n_sampled_features -- the number of features for each node to randomly look at
        chosen_inputs -- the indices of datapoints available to the tree node
        chosen_features -- the indices of variables available to the tree node
        depth -- the maximum depth of each tree
        min_leaf_size -- the minimum number of observations needed to split
        weight -- ignore
        m_ind -- list with indices of variables with missing values
        b_ind -- list with indices of binary variables
        decay -- ignore
        entropy -- the type of entropy to use (one|two|three|many|full)
        optimize -- ignore
        use_missing -- ignore
        avoid -- specify "hard" to avoid choosing variables with missing values in some levels
        layers -- number of tree depth levels to apply chosen avoid policy in
        quad -- ignore
        override -- ignore
        rng -- a random number generator
        imputer -- ignore
        root -- the root of the tree (initially None)
        parent -- the parent of the node (None for a root)
        forest -- the forest that a tree belongs to
        root_index -- nothing, used to parallelize
        mode -- used for implementing 4-way splits
        avoid_binary -- whether to apply avoid policy to binary variables as well
        """

        self.X = X
        self.n_sampled_features = n_sampled_features
        if chosen_inputs is None:
            self.chosen_inputs = np.arange[X.shape[0]]
        else:
            self.chosen_inputs = chosen_inputs
        
        self.mode = mode

        if root is None:
            self.root = self
            self.rng = rng
            self.parent = None

            self.imputed = imputed
            self.imputer = imputer

            self.m_ind = m_ind
            self.b_ind = b_ind
            
            # Create the adjacency list
            self.Al = [[]]
            # Create the leaf list to speed up getting distances
            self.Ll = []

            # Will use this to access shared data
            self.forest = forest
            
            self.weighted = forest.weighted
            self.weight = 2 ** (depth - 1)

            self.missing_profile = forest.missing_profile

            self.optimize = optimize
            self.use_missing = use_missing
            if entropy == "spectral":
                self.H = H_spectral
            elif entropy == "full":
                self.H = H_full_dim
            elif entropy == "two":
                self.H = H_two
            elif entropy == "two_even":
                self.H = H_two_even
            elif entropy == "three":
                self.H = H_three
            elif entropy == "three_even":
                self.H = H_three_even
            elif entropy == "many":
                self.H = H_many
            else:
                self.H = H_one

            if avoid is None:
                self.prob = None
            elif avoid == "soft" or avoid == "mix" or avoid == "mix2":
                self.prob = np.count_nonzero(~np.isnan(self.X), axis=0) # Probability that each variable is not missing
                p_sum = np.sum(self.prob)
                if p_sum == 0:
                    self.prob = None
                else:
                    self.prob = self.prob / p_sum
            else:
                self.prob = None

            self.avoid = avoid
            self.avoid_binary = avoid_binary
            self.layers = layers
            self.quad = quad

            UnsupervisedTree.index_counter = 1
            self.index = 0
        else:
            self.root = root
            self.parent = parent

            self.weight = weight

            self.index = UnsupervisedTree.index_counter
            UnsupervisedTree.index_counter += 1
        
        self.chosen_features = chosen_features
        self.depth = depth
        self.min_leaf_size = min_leaf_size
        self.decay = decay

        if parent is not None:
            # Update the adjacency list
            if not self.root.weighted:
                weight = 1
            if override is not None:
                weight = override
            
            self.root.Al[self.index].append((parent.index, weight))
            self.root.Al[parent.index].append((self.index, weight))

        # Score is to be maximized by splits
        if self.root.optimize == "max":
            self.score = np.NINF
        # Experimental alternative objective
        elif self.root.optimize == "min":
            self.score = np.inf

        # This step fits the tree
        self.find_split()
    
    def find_split(self):
        """
        Finds the best split for the tree by checking all splits over all variables.
        """
        replace = False
        if self.mode == 1:
            if self.depth <= 0:
                # Do not split the data if depth exhausted
                self.root.Ll.append(self.index)
                return

            for var_index in self.chosen_features:
                # Look for the best decision (if any are possible)
                self.find_better_split(var_index)
            
            if self.score == np.NINF or self.score == np.inf:
                # Do not split the data if decisions exhausted
                self.root.Ll.append(self.index)
                return
                
            split_column = self.split_column() # This is the whole column, not just the feature index

            if self.root.m_ind[self.split_feature] == 1:
                where_missing = np.nonzero(np.isnan(split_column))[0]
                not_missing = np.nonzero(~np.isnan(split_column))[0]

                # Add new nodes to the adjacency list
                self.root.Al.append([])
                self.root.Al.append([])

                m_prob = np.ones(shape=self.X.shape[1])
                mask = (self.root.m_ind == 1) | (self.root.b_ind == 1)
                m_prob[mask] = 0
                p_sum = np.sum(m_prob)
                if p_sum == 0:
                    m_prob = None
                else:
                    m_prob = m_prob / p_sum

                left_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=m_prob, replace=replace))
                right_branch_features = left_branch_features

                self.left = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[where_missing],
                                                left_branch_features, self.depth, self.min_leaf_size,
                                                self.weight, self.decay, root=self.root, parent=self, override=0,
                                                mode=2)

                self.right = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[not_missing],
                                                right_branch_features, self.depth, self.min_leaf_size,
                                                self.weight, self.decay, root=self.root, parent=self, override=0,
                                                mode=2)
            elif self.root.b_ind[self.split_feature] == 1:
                where_zero = np.nonzero(split_column == 0)[0]
                where_one = np.nonzero(split_column != 0)[0]

                # Add new nodes to the adjacency list
                self.root.Al.append([])
                self.root.Al.append([])

                m_prob = np.ones(shape=self.X.shape[1])
                mask = (self.root.m_ind == 1) | (self.root.b_ind == 1)
                m_prob[mask] = 0
                p_sum = np.sum(m_prob)
                if p_sum == 0:
                    m_prob = None
                else:
                    m_prob = m_prob / p_sum

                left_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=m_prob, replace=replace))
                right_branch_features = left_branch_features

                self.left = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[where_zero],
                                                left_branch_features, self.depth, self.min_leaf_size,
                                                self.weight, self.decay, root=self.root, parent=self, override=0,
                                                mode=2)

                self.right = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[where_one],
                                                right_branch_features, self.depth, self.min_leaf_size,
                                                self.weight, self.decay, root=self.root, parent=self, override=0,
                                                mode=2)
            else:
                where_low = np.nonzero(split_column <= self.threshold)[0] # Indices in the subset
                where_high = np.nonzero(split_column > self.threshold)[0]
        
                # Add new nodes to the adjacency list
                self.root.Al.append([])
                self.root.Al.append([])

                prob = self.root.prob
                if self.root.avoid is not None and (self.root.avoid == "hard" or self.root.avoid == "mix") and self.depth >= self.root.depth - (self.root.layers - 1):
                #if self.parent is not None and self.root.avoid is not None and (self.root.avoid == "hard2" or self.root.avoid == "mix2") and self.depth == self.root.depth - 1:
                    prob = np.ones(shape=self.X.shape[1])
                    mask = (self.root.m_ind == 1)
                    if self.root.avoid_binary:
                        mask = mask | (self.root.b_ind == 1)
                    prob[mask] = 0
                    p_sum = np.sum(prob)
                    if p_sum == 0:
                        prob = None
                    else:
                        prob = prob / p_sum

                # Randomly choose the features for each branch
                left_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=replace))
                right_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=replace))

                self.left = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[where_low], left_branch_features, 
                                            self.depth - 1, self.min_leaf_size, self.weight / 2,
                                            self.decay, root=self.root, parent=self, mode=1)
                self.right = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[where_high], right_branch_features, 
                                            self.depth - 1, self.min_leaf_size, self.weight / 2,
                                            self.decay, root=self.root, parent=self, mode=1)
        elif self.mode == 2:
            if self.depth <= 0:
                # Do not split the data if depth exhausted
                self.root.Ll.append(self.index)
                return

            for var_index in self.chosen_features:
                # Look for the best decision (if any are possible)
                self.find_better_split(var_index)
            
            if self.score == np.NINF or self.score == np.inf:
                # Do not split the data if decisions exhausted
                self.root.Ll.append(self.index)
                return
                
            split_column = self.split_column() # This is the whole column, not just the feature index

            where_low = np.nonzero(split_column <= self.threshold)[0] # Indices in the subset
            where_high = np.nonzero(split_column > self.threshold)[0]
    
            # Add new nodes to the adjacency list
            self.root.Al.append([])
            self.root.Al.append([])

            prob = self.root.prob
            if self.root.avoid is not None and (self.root.avoid == "hard" or self.root.avoid == "mix") and self.depth >= self.root.depth - (self.root.layers - 1):
            #if self.parent is not None and self.root.avoid is not None and (self.root.avoid == "hard2" or self.root.avoid == "mix2") and self.depth == self.root.depth - 1:
                prob = np.ones(shape=self.X.shape[1])
                mask = (self.root.m_ind == 1)
                if self.root.avoid_binary:
                    mask = mask | (self.root.b_ind == 1)
                prob[mask] = 0
                p_sum = np.sum(prob)
                if p_sum == 0:
                    prob = None
                else:
                    prob = prob / p_sum

            # Randomly choose the features for each branch
            left_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=replace))
            right_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=replace))

            self.left = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[where_low], left_branch_features, 
                                        self.depth - 1, self.min_leaf_size, self.weight / 2,
                                        self.decay, root=self.root, parent=self, mode=1)
            self.right = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[where_high], right_branch_features, 
                                        self.depth - 1, self.min_leaf_size, self.weight / 2,
                                        self.decay, root=self.root, parent=self, mode=1)

    def find_better_split(self, index):
        """
        Find the best split for the chosen variable. Tries all the values seen in the data to find one
        that leads to the biggest information gain when split on.
        
        Arguments:
        index -- the variable to split in
        """

        # Make a sorted list of values in this variable and get rid of missingness
        col = self.X[self.chosen_inputs, index].reshape(-1)
        n_total = len(col)
        order = np.argsort(col)
        n_missing = np.count_nonzero(np.isnan(col))

        n_complete = n_total - n_missing
        if n_complete < 2 * self.min_leaf_size:
            return
        
        if self.root.H == H_two or self.root.H == H_two_even:
            vars = np.arange(self.X.shape[1])
            vars = np.delete(vars, index)
            var2 = self.root.rng.choice(vars)

            indices = np.block([index, var2])
        elif self.root.H == H_three or self.root.H == H_three_even:
            vars = np.arange(self.X.shape[1])
            vars = np.delete(vars, index)
            var23 = self.root.rng.choice(vars, size=2, replace=False)

            indices = np.block([index, var23])
        elif self.root.H == H_full_dim:
            indices = np.arange(self.X.shape[1])
        else:
            indices = index

        H_full = self.root.H(self.X[self.chosen_inputs], order, var=indices, imputed=self.root.imputed[self.chosen_inputs],
                             use_missing=self.root.use_missing)

        if self.root.optimize == "max" and H_full <= self.score:
            # Then we will not get a higher information gain with this variable
            return
        start_j = self.min_leaf_size
        X_ordered = col[order]

        for j in range(start_j, n_complete - 1 - self.min_leaf_size):
            if X_ordered[j] == X_ordered[j+1]:
                # We do not want to calculate scores for impossible splits
                # A split must put all instances of equal values on one side
                continue
            
            x_j = X_ordered[j]
            
            # Calculate entropies of resulting distributions
            H_low = self.root.H(self.X[self.chosen_inputs], order[:j], var=indices, imputed=self.root.imputed[self.chosen_inputs],
                                use_missing=self.root.use_missing)
            H_high = self.root.H(self.X[self.chosen_inputs], order[j:], var=indices, imputed=self.root.imputed[self.chosen_inputs],
                                use_missing=self.root.use_missing)

            # We want to maximize information gain I = H(input_distribution) - |n_low|/|n_tot| H(low) - |n_high|/|n_tot| H(high)
            n_div = n_complete + n_missing * self.root.use_missing
            H_splits = (j / n_div) * H_low + (1 - j / n_div) * H_high

            if self.root.optimize == "max":
                score = H_full - H_splits
                if score > self.score:
                    self.split_feature = index
                    self.score = score
                    self.threshold = x_j
            elif self.root.optimize == "min":
                score = H_splits
                if score < self.score:
                    self.split_feature = index
                    self.score = score
                    self.threshold = x_j

    def is_leaf(self):
        """
        Checks if we reached a leaf.
        
        Return:
        True or False
        """
        return self.score == np.NINF or self.score == np.inf or self.depth <= 0

    def split_column(self):
        """
        Take the column with the variable we split over.
        
        Return:
        list of values in this variable
        """
        return self.X[self.chosen_inputs, self.split_feature]

    def apply(self, X):
        """
        Apply tree to a matrix of observations.
        
        Arguments:
        X -- the data matrix
        
        Return:
        the list of leaf assignments for each datapoint
        """
        return np.array([self.apply_row(x_i) for x_i in X])

    def apply_row(self, x_i):
        """
        Recursively apply tree to an observation. If the feature we split on is missing,
        go to the designated child node, else split by comparing with threshold value.
        
        Arguments:
        x_i -- a datapoint
        
        Return:
        its leaf assignment
        """

        if self.is_leaf():
            # Recursion base case. Say which leaf the observation ends in.
            return self.index
        
        if self.root.m_ind[self.split_feature] == 1:
            is_missing = x_i[self.split_feature] is None or np.isnan(x_i[self.split_feature])

            if is_missing:
                t = self.left
            else:
                t = self.right
        elif self.root.b_ind[self.split_feature] == 1:
            if x_i[self.split_feature] == 0:
                t = self.left
            else:
                t = self.right
        else:
            if x_i[self.split_feature] <= self.threshold:
                t = self.left
            else:
                t = self.right
        
        # Traverse the tree
        return t.apply_row(x_i)
    
    def adjacency(self, index=0):
        """
        Return the adjacency list of the tree.
        
        Arguments:
        index -- nothing, used to parallelize
        
        Return:
        an adjacency list
        """
        return self.root.Al

    def leaves(self, index=0):
        """
        Return the list of leaves of this tree.
        
        Arguments:
        index -- nothing, used to parallelize
        
        Return:
        the list of indices (in the adjacency list) of terminal nodes
        """
        return self.root.Ll

    def draw(self, index=0):
        """
        A simple way to visualize this tree.
        """

        M = adjacency_matrix_from_list(self.root.Al)
        G = nx.from_numpy_matrix(M)

        pos = nx.drawing.nx_pydot.graphviz_layout(G, "dot", root=0)
        nx.draw_networkx(G, pos=pos, node_size=20, font_size=0)

    def wasserstein(self, p, q):
        """
        Computes the tree-Wasserstein metric for the given cohorts.

        Arguments:
        p -- a cohort
        q -- another cohort
        
        Return:
        the tree-Wasserstein distance between them
        """

        n_p = p.shape[0]
        n_q = q.shape[0]

        V = len(self.root.Al)
        p_results = np.zeros(shape=(V, 3))
        q_results = np.zeros(shape=(V, 3))

        for p_i in p:
            self.apply_wasserstein(p_i, p_results)

        for q_i in q:
            self.apply_wasserstein(q_i, q_results)

        diffs = np.empty_like(p_results)
        diffs[:, 0] = p_results[:, 0] / n_p - q_results[:, 0] / n_q # probability differences
        diffs[:, 1] = p_results[:, 1] # Variable indices
        diffs[:, 2] = p_results[:, 2] # Splitting thresholds - not relevant for variable importances
        diffs = diffs[1:, :2] # We discard the root because 100% of both cohorts is under it

        importances = np.zeros(shape=self.X.shape[1]) # Make an array for the importances of all the variables
        for row in diffs:
            importances[int(row[1])] += np.abs(row[0])

        W = self.wasserstein_bfs(diffs)
        return [W, importances]

    def apply_wasserstein(self, x_i, x_results):
        """
        Classifies the observation and assigns it to each subtree it enters,
        recursively traversing the whole tree.
        
        Arguments:
        x_i -- a datapoint
        x_results -- an array to save results in
        """

        x_results[self.index, 0] += 1 # Keep counting observations in this subtree

        if self.root == self: # We will discard this later, but we need to avoid some errors
            x_results[self.index, 1] = None
            x_results[self.index, 2] = None
        else:
            x_results[self.index, 1] = self.parent.split_feature # The parent node is the cause of any difference here
            x_results[self.index, 2] = self.parent.threshold

        if self.is_leaf():
            # Recursion base case.
            return
        
        if self.root.m_ind[self.split_feature] == 1:
            is_missing = x_i[self.split_feature] is None or np.isnan(x_i[self.split_feature])

            if is_missing:
                t = self.left
            else:
                t = self.right
        elif self.root.b_ind[self.split_feature] == 1:
            if x_i[self.split_feature] == 0:
                t = self.left
            else:
                t = self.right
        else:
            if x_i[self.split_feature] <= self.threshold:
                t = self.left
            else:
                t = self.right
        
        # Traverse the tree
        t.apply_wasserstein(x_i, x_results)
        return

    def wasserstein_bfs(self, diffs):
        """
        Visits each edge of the tree starting from its root to compute
        the tree-Wasserstein metric.
        
        Arguments:
        diffs -- an array with differences for each node
        
        Return:
        the tree-Wasserstein distance
        """

        Al = self.root.Al
        n = len(Al)
        acc = 0

        # Initialize a queue and push the root onto it
        queue = deque()
        queue.append(0)

        # Keep track of what nodes we visited in this search
        visited = np.zeros(n)
        # We should not come back to where we started
        visited[0] = 1

        # Keep searching until we run out of unseen nodes
        while queue:
            # Look at the node we discovered earliest
            curr = queue.popleft()

            # Check all neighbors
            for j, w in Al[curr]:
                # To see if they were not seen before
                if not visited[j]:
                    # Then mark the node as visited
                    visited[j] = 1

                    # See how many points from p and q are in the subtree rooted at the child node
                    acc += w * np.abs(diffs[j - 1, 0])

                    # Add j to the queue so that we can visit its neighbors later
                    queue.append(j)
        
        return acc


def impute_parallel(data, seed, idx=0):
    """
    UNUSED -- ignore
    """
    return IterativeImputer(sample_posterior=True, random_state=seed).fit_transform(data)
