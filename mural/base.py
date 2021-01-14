# MURAL code
import time
import numpy as np
from collections import deque
import pickle
import json
import networkx as nx
import matplotlib.pyplot as plt

# Based on the structure proposed by Vaibhav Kumar (https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249)
# Which is derived from the fast.ai course (using the Apache license)

class UnsupervisedForest():
    """
    Realization of the random forest classifier for the unsupervised setting with samples involving missingness.
    """
    def __init__(self, X, n_estimators, n_sampled_features, batch_size, depth, min_leaf_size, decay=None):
        """
        Create and fit a random forest for unsupervised learning.
        @param X data matrix to fit to
        @param n_estimators number of trees
        @param n_sampled_features number of features that each node chooses one from, alternatively
        log base 2 ("log") or square root ("sqrt") of the total number of features
        @param batch_size number of observations each tree fits on, if None, then all observations are used
        @param depth the maximum height of the trees
        @param min_leaf_size the minimum number of observations (without missingness) needed for a separate leaf
        @param decay the factor by which the weight decays for missing value nodes
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
            assert(n_sampled_features <= X.shape[1])
            self.n_sampled_features = n_sampled_features

        self.X = X
        self.n_estimators = n_estimators
        self.depth = depth
        self.min_leaf_size = min_leaf_size

        if decay is None:
            self.decay = 0.5
        else:
            self.decay = decay

        self.rng = np.random.default_rng()
        # This step creates the forest
        self.trees = [self.create_tree() for i in range(n_estimators)]
        
        toc = time.perf_counter()
        self.time_used = toc - tic
        print(f"Completed the Unsupervised RF in {self.time_used:0.4f} seconds")
        
    def create_tree(self):
        """
        Create and fit a decision tree to be used by the forest.
        """

        chosen_features = np.sort(self.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))
        chosen_inputs = np.sort(self.rng.choice(self.X.shape[0], size=self.batch_size, replace=False))

        return UnsupervisedTree(self.X, self.n_sampled_features, chosen_inputs,
                                chosen_features, depth=self.depth, min_leaf_size=self.min_leaf_size,
                                weight=0, decay=self.decay, rng=self.rng)
    
    def apply(self, x):
        """
        Apply the fitted model to data.
        """
        return np.array([tree.apply(x) for tree in self.trees])
    
    def adjacency(self):
        """
        Get adjacency lists from each tree in a fitted model.
        """
        return [tree.adjacency() for tree in self.trees]

    def leaves(self):
        """
        Get lists of leaves for each tree in a fitted model.
        """
        return [tree.leaves() for tree in self.trees]

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
        @return an estimate of the Wasserstein distance between them
        """

        S = sum([tree.wasserstein(p, q) for tree in self.trees])
        return S / len(self.trees)

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
                 min_leaf_size, weight, decay, root=None, parent=None, rng=None):
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
        @param rng a random number generator
        """

        self.X = X
        self.n_sampled_features = n_sampled_features
        if chosen_inputs is None:
            self.chosen_inputs = np.arange[X.shape[0]]
        else:
            self.chosen_inputs = chosen_inputs

        if root is None:
            self.root = self
            
            # Create the adjacency list
            self.Al = [[]]
            # Create the leaf list to speed up getting distances
            self.Ll = []

            # Link to the forest's RNG
            self.rng = rng
            
            self.weight = 2 ** depth

            UnsupervisedTree.index_counter = 1
            self.index = 0
        else:
            self.root = root

            self.weight = weight

            self.index = UnsupervisedTree.index_counter
            UnsupervisedTree.index_counter += 1
        
        self.chosen_features = chosen_features
        self.depth = depth
        self.min_leaf_size = min_leaf_size
        self.decay = decay

        if parent is not None:
            # Update the adjacency list
            self.root.Al[self.index].append((self.parent.index, weight))
            self.root.Al[self.parent.index].append((self.index, weight))

        # Score is to be maximized by splits
        self.score = np.NINF

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
        
        if self.score == np.NINF:
            # Do not split the data if decisions exhausted
            self.root.Ll.append(self.index)
            return
              
        split_column = self.split_column()
        where_missing = np.nonzero(np.isnan(split_column))[0] # Indices in self.chosen_inputs of data with NaNs

        not_missing = np.nonzero(~np.isnan(split_column))[0]
        among_chosen = self.chosen_inputs[not_missing] # Subset of chosen_inputs, row indices in X
        complete_cases = self.X[among_chosen, self.split_feature]

        where_low = np.nonzero(complete_cases <= self.threshold)[0] # Indices in the subset
        where_high = np.nonzero(complete_cases > self.threshold)[0]
        
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
        self.missing = UnsupervisedTree(self.X, self.n_sampled_features, self.chosen_inputs[where_missing], m_branch_features, 
                                        self.depth - 1, self.min_leaf_size, self.weight * self.decay,
                                        self.decay, root=self.root, parent=self)
        self.low = UnsupervisedTree(self.X, self.n_sampled_features, among_chosen[where_low], l_branch_features, 
                                    self.depth - 1, self.min_leaf_size, self.weight / 2,
                                    self.decay, root=self.root, parent=self)
        self.high = UnsupervisedTree(self.X, self.n_sampled_features, among_chosen[where_high], h_branch_features, 
                                     self.depth - 1, self.min_leaf_size, self.weight / 2,
                                     self.decay, root=self.root, parent=self)

    def find_better_split(self, index):
        """
        Find the best split for the chosen variable. Tries all the values seen in the data to find one
        that leads to the biggest information gain when split on.
        """

        # Make a sorted list of values in this variable and get rid of missingness
        X_sorted = np.sort(self.X[self.chosen_inputs, index]).reshape(-1)
        X_sorted = X_sorted[~np.isnan(X_sorted)]
        
        # Sort into bins for the array based on values
        total_bins = np.histogram_bin_edges(X_sorted, bins="auto")

        dist_full = np.histogram(X_sorted, bins=total_bins, density=True)[0]   
        H_full = -1 * np.sum(dist_full * np.log(dist_full))
        if H_full <= self.score:
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
            H_low = -1 * np.sum(dist_low * np.log(dist_low))
            H_high = -1 * np.sum(dist_high * np.log(dist_high))

            # We want to maximize information gain I = H(input_distribution) - |n_low|/|n_tot| H(low) - |n_high|/|n_tot| H(high)
            score = H_full - (j / X_sorted.shape[0]) * H_low - (1 - j / X_sorted.shape[0]) * H_high
            if score > self.score:
                self.split_feature = index
                self.score = score
                self.threshold = x_j

    def is_leaf(self):
        """
        Checks if we reached a leaf.
        """
        return self.score == np.NINF or self.depth <= 0

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
        elif x_i[self.split_feature] is None or np.isnan(x_i[self.split_feature]):
            t = self.missing
        elif x_i[self.split_feature] <= self.threshold:
            t = self.low
        else:
            t = self.high
        
        # Traverse the tree
        return t.apply_row(x_i)
    
    def adjacency(self):
        """
        Return the adjacency list of the tree.
        """
        return self.root.Al

    def leaves(self):
        """
        Return the list of leaves of this tree.
        """
        return self.root.Ll

    def draw(self):
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
        p_results = np.zeros(V)
        q_results = np.zeros(V)

        for p_i in p:
            self.apply_wasserstein(p_i, p_results)

        for q_i in q:
            self.apply_wasserstein(q_i, q_results)

        W = self.wasserstein_bfs(p_results, n_p, q_results, n_q)
        return W

    def apply_wasserstein(self, x_i, x_results):
        """
        Classifies the observation and assigns it to each subtree it enters,
        recursively traversing the whole tree.
        """

        x_results[self.index] += 1

        if self.is_leaf():
            # Recursion base case.
            return
        elif x_i[self.split_feature] is None or np.isnan(x_i[self.split_feature]):
            t = self.missing
        elif x_i[self.split_feature] <= self.threshold:
            t = self.low
        else:
            t = self.high
        
        # Traverse the tree
        t.apply_wasserstein(x_i, x_results)
        return

    def wasserstein_bfs(self, p_results, n_p, q_results, n_q):
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
                    acc += w * np.abs((p_results[j] / n_p) - (q_results[j] / n_q))

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


def adjacency_to_distances(Al, Ll=None):
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

    dist = np.full((n, n), np.Infinity)
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
