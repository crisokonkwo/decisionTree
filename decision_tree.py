import numpy as np
from scipy import stats

# vanilla decision tree classifier that uses the provided 45 linear-regression values as features for MNIST.
class DecisionTree:
    # Node subclass to handle the graph structure
    class Node:
        # internal class for tree nodes
        def __init__(self, indices):
            self.prediction = None     # predicted class for leaf nodes
            self.feature_index = None  # feature index to split on
            self.threshold = None      # threshold value to split on
            self.l_child = None       # left child node
            self.r_child = None       # right child node
            self.indices = indices  # indices of samples that reach this node
            self.best_split = None       # cached best split info during training


    def __init__(self, num_classes, num_leaves): # initialize the decision tree with number of classes and leaves
        self.num_classes = num_classes # number of unique class labels
        self.num_leaves = num_leaves # maximum number of leaves in the tree
        self.tree_root = None # root node of the decision tree


    def best_split(self, f_column, Y):
        """
        Find the best threshold to split on for a single feature column.
        Returns (best_threshold, best_loss_after_split, left_pred, right_pred) that minimizes total 0-1 loss
        If no split improves the loss, returns (None, leaf_loss, leaf_pred, leaf_pred).
        """
        n_samples = Y.size

        # if no samples, return None
        if n_samples == 0:
            return None, float('inf'), None, None

        # Sort feature values and corresponding labels
        sort_indices = np.argsort(f_column)
        sorted_features = f_column[sort_indices]
        sorted_labels = Y[sort_indices]
        
        # If all feature values equal, no split is possible
        if sorted_features[0] == sorted_features[-1]:
            return None, np.inf, None, None

        best_threshold = None
        best_loss = np.inf
        best_left_pred = None
        best_right_pred = None

        # Initialize counts for left and right splits
        left_counts = np.zeros(self.num_classes).astype(int)
        right_counts = np.zeros(self.num_classes).astype(int)
        for label in sorted_labels:
            right_counts[label] += 1

        for i in range(n_samples - 1):
            # print("Evaluating split at index: {}".format(i))
            label = sorted_labels[i]
            left_counts[label] += 1
            right_counts[label] -= 1

            # Skip identical feature values
            if sorted_features[i] == sorted_features[i + 1]:
                continue  

            left_n = i + 1
            right_n = n_samples - left_n

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
                best_threshold = 0.5 * (sorted_features[i] + sorted_features[i + 1])
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
        Y_node = Y[indices].astype(int)
        leaf_pred, count = stats.mode(Y_node, keepdims=True).mode[0], stats.mode(Y_node, keepdims=True).count[0]
        leaf_loss = np.sum(Y_node != leaf_pred)
        print("Current prediction at node: {} and count: {} with loss: {}".format(leaf_pred, count, leaf_loss))

        best = {
            "feature": None,
            "threshold": None,
            "loss": leaf_loss,
            "left_pred": None,
            "right_pred": None,
        }

        X_node = X[indices]
        n_features = X_node.shape[1]

        for f in range(n_features):
            # print("Evaluating feature index: {}".format(feature_index))
            thresh, loss_after, l_pred, r_pred = self.best_split(X_node[:, f], Y_node) # find best split for this feature
            if loss_after < best["loss"]:
                best["feature"] = f
                best["threshold"] = thresh
                best["loss"] = loss_after
                best["left_pred"] = l_pred
                best["right_pred"] = r_pred

        return best["feature"], best["threshold"], best["loss"], best["left_pred"], best["right_pred"], leaf_loss, leaf_pred


    def train(self, features, labels, num_leaves=None):
        # Train a vanilla decision tree using greedy best-first splitting to minimize 0-1 loss
        X = np.asarray(features).astype(float)
        Y = np.asarray(labels).astype(int)
        n_samples, n_features = X.shape
        if num_leaves is None:
            num_leaves = self.num_leaves
        else:
            num_leaves = int(num_leaves)
            self.num_leaves = num_leaves

        print("[Train] decision tree with samples={}, features={}, and leaves={}".format(n_samples, n_features, num_leaves))
        
        # Root node prediction
        self.tree_root = DecisionTree.Node(indices=np.arange(n_samples))
        root_y = Y[self.tree_root.indices]
        self.tree_root.prediction, root_count = stats.mode(root_y, keepdims=True).mode[0], stats.mode(root_y, keepdims=True).count[0]
        root_loss = np.sum(root_y != self.tree_root.prediction)
        print("[Root] Node prediction={} with count={} and loss={}".format(self.tree_root.prediction, root_count, root_loss))

        leaves = [self.tree_root]  # list of current leaf nodes

        # Compute and attach best split info to a leaf
        def compute_node_split(leaf):
            feat, thresh, split_loss, l_pred, r_pred, loss_before, leaf_pred = self.split_node(leaf.indices, X, Y)
            leaf.best_split = {
                "feature": feat, "threshold": thresh,
                "loss_after": split_loss, "loss_before": loss_before,
                "l_pred": l_pred, "r_pred": r_pred, "leaf_pred": leaf_pred
            }
                
        # Precompute best split for root
        compute_node_split(self.tree_root)
        print("[Root] best split={} with indices count={}".format(self.tree_root.best_split, len(self.tree_root.indices)))

        # Grow the tree by greedily adding splits to the leaf that most reduces the overall loss until reaching the specified number of leaves or no further splits improve the loss
        print("Growing the tree...")
        round_id = 0
        while len(leaves) < max(1, num_leaves):
            round_id += 1            
            # compute/refresh best splits and collect improvements -- the leaf with the best (lowest) loss after split
            candidates = []
            for leaf in leaves:
                print("Evaluating leaf with indices count: {} and prediction: {}".format(len(leaf.indices), leaf.prediction))
                if leaf.indices is None or leaf.indices.size == 0:
                    continue
                compute_node_split(leaf)
                bs = leaf.best_split
                loss_reduction = bs["loss_before"] - bs["loss_after"] # positive gain means loss is reduced
                if bs["feature"] is not None and loss_reduction > 0:
                    candidates.append((loss_reduction, leaf))

            if not candidates:
                print(f"[Iter {round_id}] No improving splits remain. Stop.")
                break
            else:
                loss_reductions_sorted = sorted([g for g,_ in candidates], reverse=True)
                print(f"[Iter {round_id}] Leaves={len(leaves)}, improving_splits={len(candidates)}, best_gain={loss_reductions_sorted[0]}")

            # Sort loss reductions in descending order and pick the best leaf - positive gain means loss is reduced
            candidates.sort(reverse=True, key=lambda x: x[0])
            _, leaf_to_split = candidates[0]
            bs = leaf_to_split.best_split
            feat, thresh = bs["feature"], bs["threshold"]
            print(f"  -> Split leaf |idx|={len(leaf_to_split.indices)} on feature={feat}, threshold={thresh:.6f}, loss_before={bs['loss_before']}, loss_after={bs['loss_after']}")

            # Perform the split - partition the indices
            X_node = X[leaf_to_split.indices]
            left_mask = X_node[:, feat] <= thresh
            if left_mask.sum() == 0 or left_mask.sum() == X_node.shape[0]:
                # cannot split
                leaf_to_split.best_split = None
                print("     (degenerate split, skipping)")
                continue

            l_indices = leaf_to_split.indices[left_mask]
            r_indices = leaf_to_split.indices[X_node[:, feat] > thresh]

            # Create left and right child nodes
            l_node = DecisionTree.Node(indices=l_indices)
            r_node = DecisionTree.Node(indices=r_indices)
            
            l_node.prediction = stats.mode(Y[l_indices], keepdims=True).mode[0]
            r_node.prediction = stats.mode(Y[r_indices], keepdims=True).mode[0]
            
            leaf_to_split.feature_index = feat
            leaf_to_split.threshold = thresh
            leaf_to_split.l_child = l_node
            leaf_to_split.r_child = r_node
            leaf_to_split.indices = None  # Clear indices to save memory
            leaf_to_split.best_split = None  # Clear best split info
            
            leaves.remove(leaf_to_split)
            leaves.extend([l_node, r_node])

            print(f"     -> Left: |idx|={len(l_node.indices)}, pred={l_node.prediction} | Right: |idx|={len(r_node.indices)}, pred={r_node.prediction} | total leaves={len(leaves)}\n")
            
            # # Precompute best splits for the new leaves
            # compute_node_split(l_node)
            # compute_node_split(r_node)
            # print("\n")

        print(f"[Done] Final leaves={len(leaves)}\n")

    # Predict the class labels for the input samples
    def predict(self, X):
        X = np.asarray(X).astype(float)
        
        for x in X:
            node = self.tree_root
            while node.l_child is not None and node.r_child is not None:
                if x[node.feature_index] <= node.threshold:
                    node = node.l_child
                else:
                    node = node.r_child
            if not hasattr(self, 'predictions'):
                self.predictions = []
            self.predictions.append(node.prediction)

        return self.predictions

    def test(self, features_test, labels_test):
        X_test = np.asarray(features_test).astype(float)
        Y_test = np.asarray(labels_test).astype(int)

        print("Testing decision tree with {} samples...".format(X_test.shape[0]))

        predictions = self.predict(X_test)
        
        acc = (predictions == Y_test).mean()

        return acc
