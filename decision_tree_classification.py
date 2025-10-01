import time
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
    labels_train = np.asarray(labels_list).astype(int)
    images_test = np.asarray(images_list_test).astype(float)
    labels_test = np.asarray(labels_list_test).astype(int)

    # num_leaves = 10

    # Your code goes here
    num_leaves = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # plot_image(images_list[1200])
    
    # print initial stats
    print("Training set size: ", images_train.shape)
    print("Testing set size: ", images_test.shape)
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
    
    results = []  # (leaves, train_sec, test_acc)

    print("====== Decision Tree over different number of leaves ======")
    print(f"Threshold search method: Exact search\n")
    for leaves in num_leaves:
        print(f"\n--- Training and Testing with number of leaves = {leaves} ---")
        num_leaves = leaves
        # Train a vanilla decision tree with a fixed number of leaves
        model = DecisionTree(num_classes=len(np.unique(labels_train)), num_leaves=num_leaves)
        print("Training...")
        t_0 = time.perf_counter()
        model.train(X_train_features, labels_train)
        train_sec = time.perf_counter() - t_0
        train_min = train_sec / 60
        print(f"[Timing] training time = {train_min:.2f}min")

        print("\nTesting...")
        test_accuracy = model.test(X_test_features, labels_test)
        train_accuracy = model.test(X_train_features, labels_train)
        print("[Test] accuracy={:.2f}%".format(test_accuracy * 100))
        print("[Train] accuracy={:.2f}%".format(train_accuracy * 100))
        results.append((leaves, train_min, float(test_accuracy), float(train_accuracy)))

    # Final summary
    print("\n=== Summary: Accuracy vs. Leaves ===")
    print("  Leaves | TrainTime(min) | TestAcc(%) | TrainAcc(%)")
    print("  -------+---------------+-----------+-----------")
    for leaves, sec, acc, train_acc in results:
        print(f"  {leaves:6d} | {sec/60:.2f} | {acc*100:9.2f} | {train_acc*100:9.2f}")

    print("===================================================")
    
    leaves_vals = [r[0] for r in results]
    train_accs = [r[3] for r in results]
    test_accs  = [r[2] for r in results]
    
    plt.figure(figsize=(6,4))
    plt.plot(leaves_vals, train_accs, marker='o', label='Train Accuracy')
    plt.plot(leaves_vals, test_accs, marker='o', label='Test Accuracy')
    plt.xlabel("Number of leaves")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs. Leaves")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    out_file = "accuracy_vs_leaves.png"
    plt.savefig(out_file, dpi=150)