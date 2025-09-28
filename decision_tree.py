import numpy as np
from scipy import stats

# vanilla decision tree classifier that uses the provided 45 linear-regression values as features for MNIST.
class DecisionTree:
    # Node subclass to handle the graph structure
    class Node:
        # internal class for tree nodes
        def __init__(self, indices):
            self.indices = indices  # indices of samples that reach this node
            self.l_child = None       # left child node
            self.r_child = None       # right child node
            self.feature_index = None  # feature index to split on
            self.threshold = None      # threshold value to split on
            self.prediction = None     # predicted class for leaf nodes

    def __init__(self, num_classes, num_leaves): # initialize the decision tree with number of classes and leaves
        self.num_classes = num_classes # number of unique class labels
        self.num_leaves = num_leaves # maximum number of leaves in the tree
        self.tree_root = None # root node of the decision tree

    def predict(self, x):
        pass

    # def _split_node(self, node, X, Y):
    #     # Find the best split for the given node
    #     best_feature = None
    #     best_threshold = None
    #     best_gain = 0

    #     # Iterate over all features to find the best split
    #     for feature_index in range(X.shape[1]):
    #         thresholds = np.unique(X[node.indices, feature_index])
    #         for threshold in thresholds:
    #             # Compute information gain for the split
    #             gain = self._information_gain(node.indices, feature_index, threshold, X, Y)
    #             if gain > best_gain:
    #                 best_gain = gain
    #                 best_feature = feature_index
    #                 best_threshold = threshold

    #     # If a valid split was found, create child nodes
    #     if best_gain > 0:
    #         left_indices = node.indices[X[node.indices, best_feature] < best_threshold]
    #         right_indices = node.indices[X[node.indices, best_feature] >= best_threshold]
    #         node.feature_index = best_feature
    #         node.threshold = best_threshold
    #         node.l_child = DecisionTree.Node(indices=left_indices)
    #         node.r_child = DecisionTree.Node(indices=right_indices)
    #         self._split_node(node.l_child, X, Y)
    #         self._split_node(node.r_child, X, Y)
    #     else:
    #         # If no valid split, make this node a leaf
    #         node.prediction = self._get_majority_class(Y[node.indices])

    def best_split_for_feature(self, feature_column, Y):
        """
        Find the best threshold to split on for a single feature column.
        Returns (best_threshold, best_loss_after_split, left_pred, right_pred) that minimizes total 0-1 loss
        If no split improves the loss, returns (None, leaf_loss, leaf_pred, leaf_pred).
        """
        n_samples = len(Y)

        # if no samples, return None
        if n_samples == 0:
            return None, float('inf'), None, None

        # Sort feature values and corresponding labels
        sort_indices = np.argsort(feature_column)
        sorted_features = feature_column[sort_indices]
        sorted_labels = Y[sort_indices]
        
        # If all feature values equal, no split is possible
        if sorted_features[0] == sorted_features[-1]:
            return None, np.inf, None, None

        # Current prediction and loss at this node
        leaf_pred = stats.mode(sorted_labels).mode[0]
        leaf_loss = np.sum(sorted_labels != leaf_pred)

        best_threshold = None
        best_loss = leaf_loss
        best_left_pred = None
        best_right_pred = None

        # Initialize counts for left and right splits
        left_counts = np.zeros(self.num_classes) # count of each class in the left split
        # print("Left counts initialized: {}".format(left_counts))
        right_counts = np.bincount(sorted_labels, minlength=self.num_classes) # count of each class in the right split
        # print("Right counts initialized: {}".format(right_counts))

        for i in range(1, n_samples):
            label = sorted_labels[i - 1]
            left_counts[label] += 1
            right_counts[label] -= 1

            # Skip identical feature values
            if sorted_features[i] == sorted_features[i - 1]:
                continue  

            left_n = i
            right_n = n_samples - i

            # If either side is empty, skip this split
            if left_n == 0 or right_n == 0:
                continue
            
    # =====================Continue from here=============
            # Determine majority class and loss for left and right splits
            left_pred = np.argmax(left_counts)
            right_pred = np.argmax(right_counts)

            left_loss = left_n - left_counts[left_pred]
            right_loss = right_n - right_counts[right_pred]

            loss_after_split = left_loss + right_loss

            if loss_after_split < best_loss:
                best_loss = loss_after_split
                best_threshold = (sorted_features[i] + sorted_features[i - 1]) / 2
                best_left_pred = left_pred
                best_right_pred = right_pred

        return best_threshold, best_loss, best_left_pred, best_right_pred

    def split_node(self, indices, X, Y):
        """
        Search all features for the best split on this node.
        Returns (feature_index, threshold, loss_after_split, left_pred, right_pred).
        If no split improves the loss, returns (None, None, leaf_loss, leaf_pred, leaf_pred).
        """
        # Current prediction and loss at this node
        Y_node = Y[indices]
        leaf_pred, count = stats.mode(Y_node).mode[0], stats.mode(Y_node).count[0]
        print("Current prediction at node: {} and count: {}".format(leaf_pred, count))
        
        leaf_loss = np.sum(Y_node != leaf_pred)
        print("Current loss at node: {}".format(leaf_loss))

        best = {
            "feature": None,
            "threshold": None,
            "loss": leaf_loss,
            "left_pred": None,
            "right_pred": None,
        }

        X_node = X[indices]
        n_features = X_node.shape[1]

        for feature_index in range(n_features):
            print("Evaluating feature index: {}".format(feature_index))
            thresh, loss_after, l_pred, r_pred = self.best_split_for_feature(X_node[:, feature_index], Y_node) # find best split for this feature
            if loss_after < best["loss"]:
                best["feature"] = feature_index
                best["threshold"] = thresh
                best["loss"] = loss_after
                best["left_pred"] = l_pred
                best["right_pred"] = r_pred

        return best["feature"], best["threshold"], best["loss"], best["left_pred"], best["right_pred"], leaf_loss, leaf_pred

    def train(self, features, labels, num_leaves):
        # Train a vanilla decision tree using greedy best-first splitting to minimize 0-1 loss
        X = np.asarray(features).astype(float)
        Y = np.asarray(labels).astype(int)
        n_samples, n_features = X.shape

        print("Training decision tree with {} samples, {} features, and {} leaves...".format(n_samples, n_features, num_leaves))

        # initialize the tree structure with a single root node containing all samples
        self.tree_root = DecisionTree.Node(indices=np.arange(n_samples))
        # Compute initial prediction for the root node
        # Return the most common label and its count.
        tree_root_y = Y[self.tree_root.indices]
        self.tree_root.prediction, count = stats.mode(tree_root_y).mode[0], stats.mode(tree_root_y).count[0]
        print("Initial prediction at root node: {} and count: {}".format(self.tree_root.prediction, count))

        leaves = [self.tree_root] # list of current leaf nodes
        print("Number of leaves: ", len(leaves))

        print("Growing the tree...")
        # Compute and attach best split info to a leaf
        def compute_node_split(leaf):
            feature, thresh, split_loss, l_pred, r_pred, leaf_loss, leaf_pred = self.split_node(leaf.indices, X, Y)
            leaf._best = {
                "feature": feature, "threshold": thresh,
                "loss_after": split_loss, "loss_before": leaf_loss,
                "l_pred": l_pred, "r_pred": r_pred, "leaf_pred": leaf_pred
            }

        # Precompute best split for root
        compute_node_split(self.tree_root)

        # Grow the tree by greedily adding splits to the leaf that most reduces the overall loss until reaching the specified number of leaves or no further splits improve the loss
        # while len(leaves) < num_leaves:


        # compute the splits and grow the tree until reaching the specified number of leaves

        

        # self._split_node(self.tree_root, X, Y)
        # print("Final number of leaves: ", len(leaves))
        # print("Training complete.")

    def test(self, features_test, labels_test):
        pass

        #         if len(left_indices) == 0 or len(right_indices) == 0:
        #             continue

        #         y_left = Y[left_indices]
        #         y_right = Y[right_indices]

        #         left_pred = self._get_majority_class(y_left)
        #         right_pred = self._get_majority_class(y_right)

        #         loss_left = np.sum(y_left != left_pred)
        #         loss_right = np.sum(y_right != right_pred)

        #         loss_after_split = loss_left + loss_right

        #         if loss_after_split < best_loss_after:
        #             best_loss_after = loss_after_split
        #             best_feature = feature_index
        #             best_threshold = threshold
        #             best_left_pred = left_pred
        #             best_right_pred = right_pred

        # return best_feature, best_threshold, best_loss_after, best_left_pred, best_right_pred, leaf_loss, leaf_pred

    def train(self, features, labels, num_leaves):
        # Train a vanilla decision tree using greedy best-first splitting to minimize 0-1 loss
        X = np.asarray(features).astype(float)
        Y = np.asarray(labels).astype(int)
        n_samples, n_features = X.shape

        print("Training decision tree with {} samples, {} features, and {} leaves...".format(n_samples, n_features, num_leaves))

        # initialize the tree structure with a single root node containing all samples 
        self.tree_root = DecisionTree.Node(indices=np.arange(n_samples))
        # Compute initial prediction for the root node
        # Return the most common label and its count.
        tree_root_y = Y[self.tree_root.indices]
        self.tree_root.prediction, count = stats.mode(tree_root_y).mode[0], stats.mode(tree_root_y).count[0]
        print("Initial prediction at root node: {} and count: {}".format(self.tree_root.prediction, count))

        leaves = [self.tree_root] # list of current leaf nodes

        print("Growing the tree...")
        # Compute and attach best split info to a leaf
        def compute_node_split(leaf):
            feature, thresh, split_loss, l_pred, r_pred, leaf_loss, leaf_pred = self.split_node(leaf.indices, X, Y)
            leaf._best = {
                "feature": feature, "threshold": thresh,
                "loss_after": split_loss, "loss_before": leaf_loss,
                "l_pred": l_pred, "r_pred": r_pred, "leaf_pred": leaf_pred
            }

        # Precompute best split for root
        compute_node_split(self.tree_root)


        # compute the splits and grow the tree until reaching the specified number of leaves
        
        print("Number of leaves: ", len(leaves))
        
        # self._split_node(self.tree_root, X, Y)
        # print("Final number of leaves: ", len(leaves))
        # print("Training complete.")

    def test(self, features_test, labels_test):
        pass
