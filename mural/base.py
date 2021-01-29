# MURAL code
import time
import numpy as np
from collections import deque
import pickle
import json
import networkx as nx
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

# Based on the structure proposed by Vaibhav Kumar (https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249)
# Which is derived from the fast.ai course (using the Apache license)

EPSILON = np.finfo(float).eps

class UnsupervisedForest():
    """
    Realization of the random forest classifier for the unsupervised setting with samples involving missingness.
    """
    def __init__(self, X, n_estimators, n_sampled_features, batch_size, depth=4, min_leaf_size=2, 
                 decay=None, missing_profile=1, weighted=True, optimize="max"):
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
        self.n_estimators = n_estimators
        self.depth = depth
        self.min_leaf_size = min_leaf_size

        # Experiment with unweighted (all 1) edges
        self.weighted = weighted
        # Experiment with different objective
        assert (optimize == "max" or optimize == "min")
        self.optimize = optimize

        if decay is None:
            self.decay = 0.5
        else:
            self.decay = decay

        self.rng = np.random.default_rng()

        ss = np.random.SeedSequence(12345)
        child_seeds = ss.spawn(n_estimators)
        streams = [np.random.default_rng(s) for s in child_seeds]
        
        # This step creates the forest
        #self.trees = [self.create_tree() for i in range(n_estimators)]

        self.n_jobs = multiprocessing.cpu_count()
        self.trees = Parallel(n_jobs=self.n_jobs)(delayed(self.create_tree)(stream, i) for i, stream in enumerate(streams))
        
        toc = time.perf_counter()
        self.time_used = toc - tic
        print(f"Completed the Unsupervised RF in {self.time_used:0.4f} seconds")
        
    def create_tree(self, rng, root_index=0):
        """
        Create and fit a decision tree to be used by the forest.

        @rng a random number generator
        @param index nothing, used to parallelize
        """

        chosen_features = np.sort(rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))
        chosen_inputs = np.sort(rng.choice(self.X.shape[0], size=self.batch_size, replace=False))

        return UnsupervisedTree(self.X, self.n_sampled_features, chosen_inputs,
                                chosen_features, depth=self.depth, min_leaf_size=self.min_leaf_size,
                                weight=2 ** (self.depth - 1), rng=rng, decay=self.decay, forest=self, root_index=root_index)
    
    def apply(self, x):
        """
        Apply the fitted model to data.

        @param x the data to apply the tree to
        """

        result = Parallel(n_jobs=self.n_jobs)(delayed(tree.apply)(x) for tree in self.trees)

        return result
    
    def adjacency(self):
        """
        Get adjacency lists from each tree in a fitted model.
        """

        result = Parallel(n_jobs=self.n_jobs)(delayed(tree.adjacency)(i) for i, tree in enumerate(self.trees))

        return result

    def leaves(self):
        """
        Get lists of leaves for each tree in a fitted model.
        """

        result = Parallel(n_jobs=self.n_jobs)(delayed(tree.leaves)(i) for i, tree in enumerate(self.trees))

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
        W_list = [tree.wasserstein(p, q) for tree in self.trees]
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
    def __init__(self, X, n_sampled_features, chosen_inputs, chosen_features, depth,
                 min_leaf_size, weight, rng=None, decay=0.5, root=None, parent=None, forest=None, root_index=0):
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
            
            # Create the adjacency list
            self.Al = [[]]
            # Create the leaf list to speed up getting distances
            self.Ll = []

            # Will use this to access shared data
            self.forest = forest
            
            self.weight = 2 ** (depth - 1)

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
            if not self.root.forest.weighted:
                weight = 1
            
            self.root.Al[self.index].append((parent.index, weight))
            self.root.Al[parent.index].append((self.index, weight))

        # Score is to be maximized by splits
        if self.root.forest.optimize == "max":
            self.score = np.NINF
        # Experimental alternative objective
        elif self.root.forest.optimize == "min":
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
        profile = int(self.root.forest.missing_profile[self.split_feature] * len(where_missing))
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

        # Randomly choose the features for each branch
        m_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))
        l_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))
        h_branch_features = np.sort(self.root.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))

        # Create three subtrees for the data to go to
        # If we are using incomplete batches of data, we might need the "missing" subtree even if no missingness found in batch
        self.missing = UnsupervisedTree(self.X, self.n_sampled_features, self.chosen_inputs[to_missing], m_branch_features, 
                                        self.depth - 1, self.min_leaf_size, self.weight * self.decay,
                                        self.decay, root=self.root, parent=self)
        self.low = UnsupervisedTree(self.X, self.n_sampled_features, joint_low, l_branch_features, 
                                    self.depth - 1, self.min_leaf_size, self.weight / 2,
                                    self.decay, root=self.root, parent=self)
        self.high = UnsupervisedTree(self.X, self.n_sampled_features, joint_high, h_branch_features, 
                                     self.depth - 1, self.min_leaf_size, self.weight / 2,
                                     self.decay, root=self.root, parent=self)

    def find_better_split(self, index):
        """
        Find the best split for the chosen variable. Tries all the values seen in the data to find one
        that leads to the biggest information gain when split on.
        """

        # Should I add missing ones into the entropy??

        # Make a sorted list of values in this variable and get rid of missingness
        X_sorted = np.sort(self.X[self.chosen_inputs, index]).reshape(-1)
        X_sorted = X_sorted[~np.isnan(X_sorted)] # This should be faster the other way around...
        
        # Sort into bins for the array based on values
        total_bins = np.histogram_bin_edges(X_sorted, bins="auto")

        dist_full = np.histogram(X_sorted, bins=total_bins, density=True)[0]   
        H_full = -1 * np.sum(dist_full * np.log(dist_full + EPSILON))

        if self.root.forest.optimize == "max" and H_full <= self.score:
            # Then we will not get a higher information gain with this variable
            return

        bin_number = len(total_bins)
        #print(total_bins, bin_number)
        if bin_number > 2:
        
            lower_bound = np.min(X_sorted)
            upper_bound = np.max(X_sorted)
            width = (upper_bound - lower_bound) / bin_number
    
            list_var = []
            for low in range(bin_number - 1):
                list_var.append(lower_bound + (low + 1) * width)
            array_labels = np.digitize(X_sorted, list_var)
            # Calculate probabilities for the bins
            array_prob = np.bincount(array_labels) / np.sum(np.bincount(array_labels))
            # Cumulative sum of probabilities
            array_prob_cumsum = np.cumsum(array_prob)
            #print(array_prob_cumsum)
            # Start calculation at predetermined cutoff 0.25 cumulative probability 
            # Assumption that before this point there is no meaningful signal
            array_position = np.where(array_prob_cumsum >= 0.25)[0][0]
            value_position = total_bins[array_position]
            start_j = np.where(X_sorted>=value_position)[0][0]
        else:
            start_j = self.min_leaf_size

        for j in range(start_j, X_sorted.shape[0] - 1 - self.min_leaf_size):
            if X_sorted[j] == X_sorted[j+1]:
                # We do not want to calculate scores for impossible splits
                # A split must put all instances of equal values on one side
                continue
            
            x_j = X_sorted[j]

            # Calculate the optimal numbers of bins for the histograms
            bins_low = np.histogram_bin_edges(X_sorted[:j], bins="auto")
            bins_high = np.histogram_bin_edges(X_sorted[j:], bins="auto")

            # Estimate the distributions on each side of the split
            dist_low = np.histogram(X_sorted[:j], bins=bins_low, density=True)[0]
            dist_high = np.histogram(X_sorted[j:], bins=bins_high, density=True)[0]
            
            # Calculate Shannon entropy of the resulting distributions
            H_low = -1 * np.sum(dist_low * np.log(dist_low + EPSILON))
            H_high = -1 * np.sum(dist_high * np.log(dist_high + EPSILON))

            # We want to maximize information gain I = H(input_distribution) - |n_low|/|n_tot| H(low) - |n_high|/|n_tot| H(high)
            H_splits = (j / X_sorted.shape[0]) * H_low + (1 - j / X_sorted.shape[0]) * H_high

            if self.root.forest.optimize == "max":
                score = H_full - H_splits
                if score > self.score:
                    self.split_feature = index
                    self.score = score
                    self.threshold = x_j
            elif self.root.forest.optimize == "min":
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

        prob = self.root.forest.missing_profile[self.split_feature]
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
            importances[row[1]] += np.abs(row[0])

        W = self.wasserstein_bfs(p_results, n_p, q_results, n_q)
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

        prob = self.root.forest.missing_profile[self.split_feature]
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

        
def binary_affinity(result_list):
    """
    Calculates the average binary affinity from a list of results of each tree.
    
    @param result_list a list of "classification" results for a given sample on each tree
    @return the elementwise average of binary affinities across all the trees
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


def adjacency_matrix_from_list(Al):
    """
    Makes an adjacency matrix from an adjacency list representing the same graph
    @param Al an adjacency list (Python type)
    @return an adjacency matrix (ndarray)
    """

    n = len(Al)
    Am = np.zeros((n, n), dtype=int)

    for i in range(n):
        for v, w in Al[i]:
            Am[i, v] = w

    return Am


def adjacency_to_distances(Al, Ll=None, geometric=False):
    """
    Takes adjacency list and returns a distance matrix on its decision tree.
    Manually runs breadth-first search to avoid calling networkx methods
    and getting a dictionary intermediate.
    @param Al an adjacency list of a tree
    @param Ll a list of leaves of a tree (optional)
    @return a distance matrix on the tree
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
                    dist[i][j] = dist[i][curr] + w

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
    Compute the average distance between two observations across all decision trees in a forest.
    @param D_list a list containing the distance matrices for each tree in the given forest
    @param result_list already found list of results of each tree
    @return the matrix of average distances for the observations in this batch
    """

    assert len(D_list) == len(result_list)

    n = result_list[0].shape[0]
    M_acc = np.zeros(shape=(n,n))

    for D, idx in zip(D_list, result_list):
        # Choose M(x_i, x_j) to be the distance from leaf(x_i) to leaf(x_j)
        M_acc += D[idx][:,idx]

    return M_acc / len(result_list)


def load_pickle(path):
    """
    Loads an UnsupervisedForest object from pickle.
    """

    f = open(path, "rb")
    result = pickle.load(f)
    f.close()
    return result


def load_json(path):
    """
    Loads an UnsupervisedForest object from json file.
    """

    f = open(path, "r")
    result = json.load(f)
    f.close()
    return result
