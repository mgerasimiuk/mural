import time
import numpy as np
from numpy.core.defchararray import replace
from numpy.core.fromnumeric import size
from collections import deque

# Based on the structure proposed by Vaibhav Kumar (https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249)
# Which is derived from the fast.ai course (using the Apache license)

class UnsupervisedForest():
    """
    Realization of the random forest classifier for the unsupervised setting with samples involving missingness.
    """
    def __init__(self, X, n_estimators, n_sampled_features, batch_size, depth, min_leaf_size):
        """
        Create and fit a random forest for unsupervised learning.
        @param X data matrix to fit to
        @param n_estimators number of trees
        @param n_sampled_features number of features that each node chooses one from, alternatively
        log base 2 ("log") or square root ("sqrt") of the total number of features
        @param batch_size number of observations each tree fits on, if None, then all observations are used
        @param depth the maximum height of the trees
        @param min_leaf_size the minimum number of observations (without missingness) needed for a separate leaf
        """
        # Measure time to evaluate model efficiency
        tic = time.perf_counter()
        if batch_size==None:
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

        self.rng = np.random.default_rng()
        # This step creates the forest
        self.trees = [self.create_tree() for i in range(n_estimators)]
        
        toc = time.perf_counter()
        print(f"Completed the Unsupervised RF in {toc - tic:0.4f} seconds")
        
    def create_tree(self):
        """
        Create and fit a decision tree to be used by the forest.
        """
        chosen_features = np.sort(self.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))
        chosen_inputs = np.sort(self.rng.choice(self.X.shape[0], size=self.batch_size, replace=False))

        return UnsupervisedTree(self.X, self.n_sampled_features, chosen_inputs,
                                chosen_features, depth=self.depth, min_leaf_size=self.min_leaf_size, index=0)
    
    def apply(self, x):
        """
        Apply the fitted model to data.
        """
        return np.array([tree.apply(x) for tree in self.trees])

    def adjacency(self):
        """
        Get adjacency matrices from each tree in a fitted model.
        """
        return np.array([tree.adjacency()[:tree.root.max_index+1,:tree.root.max_index+1] for tree in self.trees])
    
    def adjacency_list(self):
        """
        Get adjacency lists from each tree in a fitted model.
        """
        return [tree.adjacency_list() for tree in self.trees]


class UnsupervisedTree():
    """
    Decision tree class for use in unsupervised learning of data involving missingness.
    """
    def __init__(self, X, n_sampled_features, chosen_inputs, chosen_features, depth, min_leaf_size, index, root=None, parent=None):
        """
        Create and fit an unsupervised decision tree.
        @param X data matrix
        @param n_sampled_features number of features to choose from at each node
        @param chosen_inputs ndarray containing the indices of chosen observations in this data matrix
        @param chosen_features ndarray containing the indices of features to choose from at each node
        @param depth maximum height of this tree
        @param min_leaf_size minimum number of observations (without missingness) needed to create a new leaf
        @param index the index of the root of this tree, where, if node j has children, they are 3j+1, 3j+2, 3j+3
        @param root the handle of the root of this tree, if None, then this tree is marked as root to all its children
        @param parent the handle of the parent of this node, if None, then this tree is marked as root
        """
        self.X = X
        self.n_sampled_features = n_sampled_features
        if chosen_inputs is None:
            self.chosen_inputs = np.arange[X.shape[0]]
        else:
            self.chosen_inputs = chosen_inputs

        if root is None:
            self.root = self
            max_leaf = int((3 ** (depth + 1) - 1) / 2)
            
            # Create the adjacency matrix
            # For more efficient code, it should be better to dynamically grow an adjacency list
            # and only turn it into a matrix in the final step
            self.A = np.zeros((max_leaf, max_leaf))
            self.Al = [[]]
            
            UnsupervisedTree.index_counter = 1
            self.index = 0
            self.max_index = 0
        else:
            self.root = root
            self.A = root.A
            self.Al = root.Al
            self.index = UnsupervisedTree.index_counter
            UnsupervisedTree.index_counter += 1
        
        self.root.max_index = self.index
        
        self.chosen_features = chosen_features
        self.depth = depth
        self.min_leaf_size = min_leaf_size
        #self.index = index

        if parent is not None:
            # Maintain a link between the node an its parent in case of need for traversal
            self.parent = parent
            
            # Update the adjacency matrix
            self.root.A[self.index, self.parent.index] = 1
            self.root.A[self.parent.index, self.index] = 1
            self.root.Al[self.index].append(self.parent.index)
            self.root.Al[self.parent.index].append(self.index)

        self.total_features = X.shape[1]
        # Score is to be minimized by splits
        self.score = np.Inf

        self.rng = np.random.default_rng()
        # This step fits the tree
        self.find_split()
    
    def find_split(self):
        """
        Find the best split for the tree by checking all splits over all variables.
        """
        for index in self.chosen_features:
            self.find_better_split(index)
        
        if self.is_leaf():
            # Do not actually split the data
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
        m_branch_features = np.sort(self.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))
        l_branch_features = np.sort(self.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))
        h_branch_features = np.sort(self.rng.choice(self.X.shape[1], size=self.n_sampled_features, replace=False))

        # Create three subtrees for the data to go to
        # If we are using incomplete batches of data, we might need the "missing" subtree even if no missingness found in batch
        self.missing = UnsupervisedTree(self.X, self.n_sampled_features, self.chosen_inputs[where_missing], m_branch_features, 
                                        self.depth - 1, self.min_leaf_size, 3 * self.index + 1, root=self.root, parent=self)
        self.low = UnsupervisedTree(self.X, self.n_sampled_features, among_chosen[where_low], l_branch_features, 
                                    self.depth - 1, self.min_leaf_size, 3 * self.index + 2, root=self.root, parent=self)
        self.high = UnsupervisedTree(self.X, self.n_sampled_features, among_chosen[where_high], h_branch_features, 
                                        self.depth - 1, self.min_leaf_size, 3 * self.index + 3, root=self.root, parent=self)

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
        bin_number = len(total_bins)
        #print(total_bins, bin_number)
        if bin_number>2:
        
            lower_bound = np.min(X_sorted)
            upper_bound = np.max(X_sorted)
            width = (upper_bound - lower_bound)/bin_number
    
            list_var = []
            for low in range(bin_number-1):
                list_var.append(lower_bound + (low+1)*width)
            array_labels = np.digitize(X_sorted, list_var)
            # Calculate probabilities for the bins
            array_prob = np.bincount(array_labels)/np.sum(np.bincount(array_labels))
            # Cumulative sum of probabilities
            array_prob_cumsum = np.cumsum(array_prob)
            #print(array_prob_cumsum)
            # Start calculation at predetermined cutoff 0.25 cumulative probability 
            # Assumption that before this point there is no meaningful signal
            array_position = np.where(array_prob_cumsum>=0.25)[0][0]
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

            # We want to maximize information gain I = H(input_distribution) - score
            # So we minimize score here and remember best threshold so far
            score = (j / X_sorted.shape[0]) * H_low + (1 - j / X_sorted.shape[0]) * H_high
            if score < self.score:
                self.split_feature = index
                self.score = score
                self.threshold = x_j

    def is_leaf(self):
        """
        Checks if we reached a leaf.
        """
        return self.score == np.Inf or self.depth <= 0

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
        Return the adjacency matrix of the tree.
        """
        return self.root.A
    
    def adjacency_list(self):
        """
        Return the adjacency list of the tree.
        """
        return self.root.Al

        
def binary_affinity(leaf_list):
    """
    Calculates the average binary affinity from a list of leaves on each tree.
    
    @param leaf_list a list of "classification" results for a given sample on each tree
    @return the elementwise average of binary affinities across all the trees
    """
    M_list = []

    for idx in leaf_list:
        n = idx.shape[0]
        I1 = np.tile(idx, [n,1])
        I2 = np.tile(idx[:,None], [1,n])
        # Force broadcasting to get a binary affinity matrix
        # (Do observations i and j have the same leaf assignment?)
        M = np.equal(I1, I2).astype("int")
        M_list.append(M)

    M_list = np.asarray(M_list)
    M_avg = np.mean(M_list, axis=0)

    return M_avg


def adjacency_to_distances(A):
    """
    Takes adjacency matrix and returns a distance matrix on its decision tree.
    Unlike method above, manually runs breadth-first search to avoid calling networkx methods
    and getting a dictionary intermediate.
    @param A an adjacency matrix of a tree or forest
    @return a list of distance matrices, one for each tree in a forest
    """

    # Make a distance matrix for each root
    # The number of nodes in one decision tree
    n = A.shape[0]

    dist_simple = np.zeros((n, n))
    dist_exp = np.full((n, n), np.Infinity)
    np.fill_diagonal(dist_exp, 0)

    # The steps below compute shortest paths by calling BFS from each node
    for i in range(0, A.shape[0]):
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

            # Check all nodes on this tree
            for j in range(0, n):
                # To see if they are adjacent to the current one and not seen before
                if A[curr][j] == 1 and not visited[j]:
                    # Then mark the node as visited
                    visited[j] = 1

                    # j's distance from node i is one more than that from i to j's predecessor (curr)
                    dist_simple[i][j] = dist_simple[i][curr] + 1
                    # Compute the exponential edge distance
                    dist_exp[i][j] = 2 ** dist_simple[i][j] - 1

                    # Add j to the queue so that we can visit its neighbors later
                    q.append(j)

    # The distance matrix for this tree is now done
    return dist_exp


def adjacency_matrix_from_list(Al):
    """
    Makes an adjacency matrix from an adjacency list representing the same graph

    @param Al an adjacency list (Python type)
    @return an adjacency matrix (ndarray)
    """
    n = len(Al)
    Am = np.zeros((n, n), dtype=int)

    for i in range(n):
        for v in Al[i]:
            Am[i, v] = 1

    return Am


def adjacency_list_to_distances(Al):
    """
    Takes adjacency matrix and returns a distance matrix on its decision tree.
    Unlike method above, manually runs breadth-first search to avoid calling networkx methods
    and getting a dictionary intermediate.

    @param Al an adjacency list of a tree
    @return a distance matrix of the tree
    """
    # The number of nodes in one decision tree
    n = len(Al)

    dist_simple = np.zeros((n, n))
    dist_exp = np.full((n, n), np.Infinity)
    np.fill_diagonal(dist_exp, 0)

    # The steps below compute shortest paths by calling BFS from each node
    for i in range(0, n):
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
            for j in Al[curr]:
                # To see if they were not seen before
                if not visited[j]:
                    # Then mark the node as visited
                    visited[j] = 1

                    # j's distance from node i is one more than that from i to j's predecessor (curr)
                    dist_simple[i][j] = dist_simple[i][curr] + 1
                    # Compute the exponential edge distance
                    dist_exp[i][j] = 2 ** dist_simple[i][j] - 1

                    # Add j to the queue so that we can visit its neighbors later
                    q.append(j)

    # The distance matrix for this tree is now done
    return dist_exp


def get_average_distance(D_list, leaf_list):
    """
    Compute the average distance between two observations across all decision trees in a forest.
    @param D_list a list containing the distance matrices for each tree in the given forest
    @param list already found list of leaves for each tree
    @return the matrix of average distances for the observations in this batch
    """

    M_list = []

    for D, idx in zip(D_list, leaf_list):
        # Choose M(x_i, x_j) to be the distance from leaf(x_i) to leaf(x_j)
        M = D[idx][:,idx]
        M_list.append(M)

    M_list = np.asarray(M_list)
    M_avg = np.mean(M_list, axis=0)

    return M_avg
