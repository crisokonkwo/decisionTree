from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from decision_tree import DecisionTree


def plot_image(image_list):
    image = np.asarray(image_list)

    plt.figure()
    plt.imshow(np.reshape(image, (28,28)), cmap='gray_r')
    plt.show()


def load_weights(file_name, num_lines, dim):
    w = np.zeros((num_lines, dim))
    b = np.zeros((num_lines, 1))

    count = 0

    with open(file_name) as file:
        for line in file:
            values = np.array([float(i) for i in line.split(',')])

            w[count, :] = values[0:dim]
            b[count] = values[dim]

            count += 1

    return (w, b)


if __name__ == "__main__":
    
    mndata = MNIST('./datasets/MNIST/raw')
    images_list, labels_list = mndata.load_training()
    images_list_test, labels_list_test = mndata.load_testing()

    # Use this only if you need to use the provided weights
    num_lines = 45
    (w, b) = load_weights('weights.txt', num_lines, 28 * 28)

    images_train = np.asarray(images_list).astype(float)
    labels_train = np.asarray(labels_list).astype(float)
    images_test = np.asarray(images_list_test).astype(float)
    labels_test = np.asarray(labels_list_test).astype(int)

    num_leaves = 10

    # Your code goes here

    # plot_image(images_list[1200])

    # print initial stats
    print("Training set size: ", images_train.shape)
    print("Testing set size: ", images_test.shape)
    print("Number of leaves: ", num_leaves)
    print("Number of classes: ", len(np.unique(labels_train)))
    print("weights shape: ", w.shape)
    print("bias shape: ", b.shape)
    print("================================")

    # compute the feature vector for each image X @ w.T + b
    X_train_features = images_train @ w.T + b.ravel()
    X_test_features = images_test @ w.T + b.ravel()
    print("Feature vector shape: ", X_train_features.shape)
    print("Feature vector test shape: ", X_test_features.shape)
    print("================================\n")

    print("Initializing the decision tree...")
    # Train a vanilla decision tree with a fixed number of leaves
    model = DecisionTree(num_classes=len(np.unique(labels_train)), num_leaves=num_leaves)

    print("Training...")
    model.train(X_train_features, labels_train, num_leaves)

