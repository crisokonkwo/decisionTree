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


    def best_split(self, feature_column, Y):
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
        leaf_pred = stats.mode(sorted_labels, keepdims=True).mode[0]
        leaf_loss = np.sum(sorted_labels != leaf_pred)
        # print("Current prediction at node: {} and loss: {}".format(leaf_pred, leaf_loss))

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
            
            # Determine majority class and loss for left and right splits
            left_pred = np.argmax(left_counts)
            right_pred = np.argmax(right_counts)
            # print("Left pred: {}, Right pred: {}".format(left_pred, right_pred))

            # Calculate 0-1 loss for left and right splits
            left_loss = left_n - left_counts[left_pred]
            right_loss = right_n - right_counts[right_pred]
            # print("Evaluated threshold: {}, Left pred: {}, Left loss: {}, Right pred: {}, Right loss: {}".format((sorted_features[i] + sorted_features[i - 1]) / 2, left_pred, left_loss, right_pred, right_loss))  

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
        leaf_pred, count = stats.mode(Y_node, keepdims=True).mode[0], stats.mode(Y_node, keepdims=True).count[0]
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
            # print("Evaluating feature index: {}".format(feature_index))
            thresh, loss_after, l_pred, r_pred = self.best_split(X_node[:, feature_index], Y_node) # find best split for this feature
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
        self.tree_root.prediction, count = stats.mode(tree_root_y, keepdims=True).mode[0], stats.mode(tree_root_y, keepdims=True).count[0]
        print("Initial prediction at root node: {} and count: {}".format(self.tree_root.prediction, count))

        leaves = [self.tree_root] # list of current leaf nodes
        print("Number of leaves: ", len(leaves))

        print("Growing the tree...")
        # Compute and attach best split info to a leaf
        def compute_node_split(leaf):
            feature, thresh, split_loss, l_pred, r_pred, leaf_loss, leaf_pred = self.split_node(leaf.indices, X, Y)
            leaf.best_split = {
                "feature": feature, "threshold": thresh,
                "loss_after": split_loss, "loss_before": leaf_loss,
                "l_pred": l_pred, "r_pred": r_pred, "leaf_pred": leaf_pred
            }
            print("Computed best split for node: ", leaf.best_split)
            # If the split improves the loss, we can use it
            # if leaf.best_split["loss_after"] < leaf.best_split["loss_before"]:
            #     leaf.split = leaf.best_split

        
        print("Debug: Before precomputing best split for root node")
        # Precompute best split for root
        compute_node_split(self.tree_root)
        print("Initial best split at root: ", self.tree_root.best_split)

        # Grow the tree by greedily adding splits to the leaf that most reduces the overall loss until reaching the specified number of leaves or no further splits improve the loss
        # compute the splits and grow the tree until reaching the specified number of leaves
        while len(leaves) < num_leaves:
            # Find the leaf with the best (lowest) loss after split

            best_leaf = None
            best_loss_reduction = 0
            for leaf in leaves:
                if hasattr(leaf, 'best_split') and leaf.best_split["feature"] is not None:
                    loss_reduction = leaf.best_split["loss_before"] - leaf.best_split["loss_after"]
                    if loss_reduction > best_loss_reduction:
                        best_loss_reduction = loss_reduction
                        best_leaf = leaf

            # If no leaf can be split to reduce loss, stop growing
            if best_leaf is None:
                print("No further splits improve the loss. Stopping growth.")
                break

            # Perform the split on the best leaf
            feature_index = best_leaf.best_split["feature"]
            threshold = best_leaf.best_split["threshold"]
            left_pred = best_leaf.best_split["l_pred"]
            right_pred = best_leaf.best_split["r_pred"]

            # Create left and right child nodes
            left_indices = best_leaf.indices[X[best_leaf.indices, feature_index] <= threshold]
            right_indices = best_leaf.indices[X[best_leaf.indices, feature_index] > threshold]

            best_leaf.l_child = DecisionTree.Node(indices=left_indices)
            best_leaf.l_child.prediction = left_pred
            best_leaf.r_child = DecisionTree.Node(indices=right_indices)
            best_leaf.r_child.prediction = right_pred

            # Update the current leaf to be an internal node
            best_leaf.feature_index = feature_index
            best_leaf.threshold = threshold
            best_leaf.prediction = None  # Internal nodes do not have predictions

            # Remove the split leaf from leaves and add its children
            leaves.remove(best_leaf)
            leaves.append(best_leaf.l_child)
            leaves.append(best_leaf.r_child)

            # Compute best splits for the new leaves
            compute_node_split(best_leaf.l_child)
            compute_node_split(best_leaf.r_child)

            print("Split on feature {} at threshold {}. New number of leaves: {}".format(feature_index, threshold, len(leaves)))

        print("Final number of leaves: ", len(leaves))
        print("Training complete.")

        # Cleanup training-only attributes to reduce memory footprint
        for leaf in leaves:
            if hasattr(leaf, 'best_split'):
                del leaf.best_split

        # Cleanup tree structure
        self.tree_root = None

    def predict(self, X):
        X = np.asarray(X).astype(float)
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            node = self.tree_root
            while node is not None:
                if node.prediction is not None:
                    predictions[i] = node.prediction
                    break
                if X[i, node.feature_index] <= node.threshold:
                    node = node.l_child
                else:
                    node = node.r_child

        return predictions

    def test(self, features_test, labels_test):
        X_test = np.asarray(features_test).astype(float)
        Y_test = np.asarray(labels_test).astype(int)

        print("Testing decision tree with {} samples...".format(X_test.shape[0]))

        # TODO: Implement the testing logic
        predictions = self.predict(X_test)

        return np.mean(predictions == Y_test)


    def test(self, features_test, labels_test):
        X_test = np.asarray(features_test).astype(float)
        Y_test = np.asarray(labels_test).astype(int)

        print("Testing decision tree with {} samples...".format(X_test.shape[0]))

        # TODO: Implement the testing logic
        predictions = self.predict(X_test)

        return np.mean(predictions == Y_test)
