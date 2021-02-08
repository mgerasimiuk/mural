# MURAL
# Written by Michal Gerasimiuk for a project undertaken with Dennis Shung
# Based on the structure proposed by Vaibhav Kumar (https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249)
# Which is derived from the fast.ai course (using the Apache license)
# Documented using javadoc because I don't know anything else ¯\_(ツ)_/¯
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
N_JOBS = min(multiprocessing.cpu_count(), 9)


class UnsupervisedForest():
    """
    Realization of the random forest classifier for the unsupervised setting with samples involving missingness.
    """
    def __init__(self, X, n_estimators, n_sampled_features, batch_size, imputed=None, depth=4, min_leaf_size=2, 
                decay=None, missing_profile=1, weighted=True, optimize="max", entropy="one", use_missing=True,
                avoid=None, layers=1, quad=False):
        """
        Create and fit a random forest for unsupervised learning.
        @param X data matrix to fit to
        @param n_estimators number of trees
        @param n_sampled_features number of features that each node chooses one from, alternatively
        log base 2 ("log") or square root ("sqrt") of the total number of features (BAD DO NOT USE!!!)
        @param batch_size number of observations each tree fits on, if None, then all observations are used
        @param depth the maximum height of the trees
        @param min_leaf_size the minimum number of observations (without missingness) needed for a separate leaf
        @param decay the factor by which the weight decays for missing value nodes
        @param missing_profile an array whose length is the number of features or {0,1}, default=1 -
        probabilities (confidence) that each feature is missing not at random
        @param weighted True (default) or False (to use equal weights)
        @param optimize "max" (default) or "min" (to use old splitting objective)
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
        self.layers = layers
        self.quad = quad

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

        ss = np.random.SeedSequence(12345)
        child_seeds = ss.spawn(n_estimators)
        streams = [np.random.default_rng(s) for s in child_seeds]
        #states = ss.generate_state(n_estimators)

        #imputers = [IterativeImputer(sample_posterior=True, random_state=state) for state in states]
        imputers = np.arange(n_estimators)
        #imputed = Parallel(n_jobs=N_JOBS)(delayed(impute_parallel)(self.X, seed, i) for i, seed in enumerate(child_seeds))
        self.trees = Parallel(n_jobs=N_JOBS)(delayed(self.create_tree)(stream, imputers[i], i) for i, stream in enumerate(streams))
        
        toc = time.perf_counter()
        self.time_used = toc - tic
        print(f"Completed the Unsupervised RF in {self.time_used:0.4f} seconds")
        
    def create_tree(self, rng, imputer, root_index=0):
        """
        Create and fit a decision tree to be used by the forest.

        @rng a random number generator
        @param imputer an imputer object
        @param index nothing, used to parallelize
        """

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
            prob[mask] = 0
            p_sum = np.sum(prob)
            if p_sum == 0:
                prob = None
            else:
                prob = prob / p_sum
        else:
            prob = None

        chosen_features = np.sort(rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=False))
        chosen_inputs = np.sort(rng.choice(self.X.shape[0], size=self.batch_size, replace=False))

        if self.imputed is None:
            #imputed = imputer.fit_transform(self.X)
            imputed = self.X # This is to be removed later
        else:
            imputed = self.imputed

        return UnsupervisedTree(self.X, imputed, self.n_sampled_features, chosen_inputs, chosen_features,
                                depth=self.depth, min_leaf_size=self.min_leaf_size, weight=2 ** (self.depth - 1),
                                decay=self.decay, entropy=self.entropy, optimize=self.optimize,
                                use_missing=self.use_missing, rng=rng, imputer=imputer, forest=self,
                                root_index=root_index, avoid=self.avoid, layers=self.layers, quad=self.quad)
    
    def apply(self, x):
        """
        Apply the fitted model to data.

        @param x the data to apply the tree to
        """

        result = Parallel(n_jobs=N_JOBS)(delayed(tree.apply)(x) for tree in self.trees)

        return result
    
    def adjacency(self):
        """
        Get adjacency lists from each tree in a fitted model.
        """

        result = Parallel(n_jobs=N_JOBS)(delayed(tree.adjacency)(i) for i, tree in enumerate(self.trees))

        return result

    def leaves(self):
        """
        Get lists of leaves for each tree in a fitted model.
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

        @param p a cohort
        @param q another cohort
        @return an estimate of the Wasserstein distance between them, array of averaged importances
        """
        W_list = np.array(Parallel(n_jobs=N_JOBS)(delayed(tree.wasserstein)(p, q) for tree in self.trees))
        W_vals = W_list[:, 0]
        W_imps = np.array(W_list[:, 1])
        return sum(W_vals) / len(self.trees), np.mean(W_imps, axis=0)

    def to_pickle(self, path):
        """
        Saves the object to a pickle with the given path.
        """

        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    def to_json(self, path):
        """
        Saves the object to a json file with the given path.
        """

        f = open(path, "w")
        json.dump(self, f)
        f.close()


class UnsupervisedTree():
    """
    Decision tree class for use in unsupervised learning of data involving missingness.
    """
    def __init__(self, X, imputed, n_sampled_features, chosen_inputs, chosen_features, depth, min_leaf_size, 
                 weight, decay=0.5, entropy="one", optimize="max", use_missing=True, avoid=None, layers=1,
                 quad=False, override=None, rng=None, imputer=None, root=None, parent=None, forest=None, 
                 root_index=0):
        """
        Create and fit an unsupervised decision tree.
        @param X data matrix
        @param n_sampled_features number of features to choose from at each node
        @param chosen_inputs ndarray containing the indices of chosen observations in this data matrix
        @param chosen_features ndarray containing the indices of features to choose from at each node
        @param depth maximum height of this tree
        @param min_leaf_size minimum number of observations (without missingness) needed to create a new leaf
        @param weight the weight of the edge in which this node is the child
        @param decay the factor by which the weight for missing value nodes decays
        @param root the handle of the root of this tree, if None, then this tree is marked as root to all its children
        @param parent the handle of the parent of this node, if None, then this tree is marked as root
        @param forest the handle of the forest that the tree belongs to
        """

        self.X = X
        self.n_sampled_features = n_sampled_features
        if chosen_inputs is None:
            self.chosen_inputs = np.arange[X.shape[0]]
        else:
            self.chosen_inputs = chosen_inputs

        if root is None:
            self.root = self
            self.rng = rng
            self.parent = None

            self.imputed = imputed
            self.imputer = imputer
            
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
        Find the best split for the tree by checking all splits over all variables.
        """

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
        where_missing = np.nonzero(np.isnan(split_column))[0] # Indices in self.chosen_inputs of data with NaNs

        # Split missing values between regular nodes and missing node based on missing_profile
        profile = int(self.root.missing_profile[self.split_feature] * len(where_missing))
        permute = self.root.rng.permutation(where_missing)
        to_missing = permute[0:profile] # These go to the missing node
        to_rest = permute[profile:] # These go to 50/50 to the other two

        # If there are observations with missingness left:
        if profile != len(where_missing):
            half = len(where_missing) // 2
            permute = self.root.rng.permutation(to_rest)
            missing_to_low = self.chosen_inputs[permute[0:half]] # Row indices in X
            missing_to_high = self.chosen_inputs[permute[half:]]
        else:
            missing_to_low = []
            missing_to_high = []

        not_missing = np.nonzero(~np.isnan(split_column))[0]
        among_chosen = self.chosen_inputs[not_missing] # Subset of chosen_inputs, row indices in X
        complete_cases = self.X[among_chosen, self.split_feature]

        where_low = np.nonzero(complete_cases <= self.threshold)[0] # Indices in the subset
        where_high = np.nonzero(complete_cases > self.threshold)[0]
    
        # Randomly apportion randomly missing values
        joint_low = np.array(np.concatenate((among_chosen[where_low], missing_to_low), axis=None), dtype=int) # Row indices in X
        joint_high = np.array(np.concatenate((among_chosen[where_high], missing_to_high), axis=None), dtype=int)
        
        # Add new nodes to the adjacency list
        self.root.Al.append([])
        self.root.Al.append([])
        self.root.Al.append([])

        prob = self.root.prob
        if self.root.avoid is not None and (self.root.avoid == "hard" or self.root.avoid == "mix") and self.depth >= self.root.depth - (self.root.layers - 1):
        #if self.parent is not None and self.root.avoid is not None and (self.root.avoid == "hard2" or self.root.avoid == "mix2") and self.depth == self.root.depth - 1:
            prob = np.ones(shape=self.X.shape[1])
            mask = np.any(np.isnan(self.X), axis=0)
            prob[mask] = 0
            p_sum = np.sum(prob)
            if p_sum == 0:
                prob = None
            else:
                prob = prob / p_sum

        if self.root.quad: # Force a different probability distribution for the replacement split in the missing case
            m_prob = np.ones(shape=self.X.shape[1])
            mask = np.any(np.isnan(self.X), axis=0)
            m_prob[mask] = 0
            p_sum = np.sum(m_prob)
            if p_sum == 0:
                m_prob = None
            else:
                m_prob = m_prob / p_sum
        else:
            m_prob = prob

        # Randomly choose the features for each branch
        m_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=m_prob, replace=False))
        l_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=False))
        h_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, p=prob, replace=False))

        # Create three subtrees for the data to go to
        # If we are using incomplete batches of data, we might need the "missing" subtree even if no missingness found in batch
        if self.root.quad: # This simulates a four-way split
            self.missing = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[to_missing],
                                            m_branch_features, self.depth, self.min_leaf_size,
                                            self.weight, self.decay, root=self.root, parent=self, override=0)
        else:
            self.missing = UnsupervisedTree(self.X, None, self.n_sampled_features, self.chosen_inputs[to_missing],
                                            m_branch_features, self.depth - 1, self.min_leaf_size,
                                            self.weight * self.decay, self.decay, root=self.root, parent=self)

        self.low = UnsupervisedTree(self.X, None, self.n_sampled_features, joint_low, l_branch_features, 
                                    self.depth - 1, self.min_leaf_size, self.weight / 2,
                                    self.decay, root=self.root, parent=self)
        self.high = UnsupervisedTree(self.X, None, self.n_sampled_features, joint_high, h_branch_features, 
                                     self.depth - 1, self.min_leaf_size, self.weight / 2,
                                     self.decay, root=self.root, parent=self)

    def find_better_split(self, index):
        """
        Find the best split for the chosen variable. Tries all the values seen in the data to find one
        that leads to the biggest information gain when split on.
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
        """
        return self.score == np.NINF or self.score == np.inf or self.depth <= 0

    def split_column(self):
        """
        Take the column with the variable we split over.
        """
        return self.X[self.chosen_inputs, self.split_feature]

    def apply(self, X):
        """
        Apply tree to a matrix of observations.
        """
        return np.array([self.apply_row(x_i) for x_i in X])

    def apply_row(self, x_i):
        """
        Recursively apply tree to an observation. If the feature we split on is missing,
        go to the designated child node, else split by comparing with threshold value.
        """

        if self.is_leaf():
            # Recursion base case. Say which leaf the observation ends in.
            return self.index
        
        is_missing = x_i[self.split_feature] is None or np.isnan(x_i[self.split_feature])

        prob = self.root.missing_profile[self.split_feature]
        rolled = self.root.rng.random() <= prob

        if is_missing and not rolled:
            auto_low = self.root.rng.choice([True, False])
        else:
            auto_low = False

        if is_missing and rolled:
            t = self.missing
        elif auto_low or x_i[self.split_feature] <= self.threshold:
            t = self.low
        else:
            t = self.high
        
        # Traverse the tree
        return t.apply_row(x_i)
    
    def adjacency(self, index=0):
        """
        Return the adjacency list of the tree.
        """
        return self.root.Al

    def leaves(self, index=0):
        """
        Return the list of leaves of this tree.
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

        @param p a cohort
        @param q another cohort
        @return the tree-Wasserstein distance between them
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
        
        is_missing = x_i[self.split_feature] is None or np.isnan(x_i[self.split_feature])

        prob = self.root.missing_profile[self.split_feature]
        rolled = self.root.rng.random() <= prob

        if is_missing and not rolled:
            auto_low = self.root.rng.choice([True, False])
        else:
            auto_low = False

        if is_missing and rolled:
            t = self.missing
        elif auto_low or x_i[self.split_feature] <= self.threshold:
            t = self.low
        else:
            t = self.high
        
        # Traverse the tree
        t.apply_wasserstein(x_i, x_results)
        return

    def wasserstein_bfs(self, diffs):
        """
        Visits each edge of the tree starting from its root to compute
        the tree-Wasserstein metric.
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
    Helper function for multiple imputation in parallel.
    """
    return IterativeImputer(sample_posterior=True, random_state=seed).fit_transform(data)
