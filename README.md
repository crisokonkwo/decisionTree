# Decision Tree Classification on MNIST Dataset

This project implements a decision tree classifier from scratch for MNIST digit classification. The decision tree uses pre-computed linear features from the MNIST dataset to classify handwritten digits.

## Features

- **Decision Tree Implementation**: Custom decision tree classifier with configurable number of leaves
- **Exact Threshold Search**: Finds optimal split thresholds by evaluating all possible splits
- **Greedy Best-First Splitting**: Uses a priority queue to efficiently select the best splits
- **MNIST Classification**: Trained on MNIST dataset using 45 pre-computed linear features
- **Performance Analysis**: Evaluates accuracy vs. number of leaves relationship

## Requirements

### Python Version

- Python 3.13 (tested with 3.13.7)

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Dataset Setup

1. **MNIST Dataset**: The project uses the MNIST dataset which should be automatically downloaded when first run.

2. **Pre-computed Features**: The `weights.txt` file contains 45 pre-computed linear regression weights that transform the raw MNIST pixels (784 dimensions) into a 45-dimensional feature space.

## How to Run

### Quick Start

To run the complete experiment with default settings:

```bash
python decision_tree_classification.py
```

This will:

1. Load the MNIST training and testing datasets
2. Transform raw pixel data using the pre-computed weights
3. Train decision trees with varying numbers of leaves (10, 20, 30, ..., 1000)
4. Evaluate accuracy on both training and test sets
5. Generate an accuracy vs. leaves plot saved as `accuracy_vs_leaves.png`

### Expected Output

The script will output training progress and results similar to:

```
Training set size:  (60000, 784)
Testing set size:  (10000, 784)
Number of classes:  10
weights shape:  (45, 784)
Feature vector shape:  (60000, 45)
================================

====== Decision Tree over different number of leaves ======
Threshold search method: Exact search

--- Training and Testing with number of leaves = 10 ---
Training...
Train decision tree with samples=60000, features=45, and leaves=10
Root Node prediction=1 with count=6742 and loss=53258
...
Timing training time = 2.45min

Testing...
Test accuracy=85.20%
Train accuracy=86.15%
```

### Customizing the Experiment

You can modify the `decision_tree_classification.py` script to:

1. **Change number of leaves**: Modify the `num_leaves` list:

   ```python
   num_leaves = [10, 20, 50, 100]  
   ```

2. **Use a single tree**: Set a specific number of leaves:

   ```python
   num_leaves = 50
   model = DecisionTree(num_classes=10, num_leaves=num_leaves)
   model.train(X_train_features, labels_train)
   ```

## Algorithm Details

### Training Process

1. Create root node with all training samples
2. For each potential leaf, find the best feature and threshold that minimizes 0-1 loss
3. Use a priority queue to always split the leaf that provides maximum loss reduction
4. Navigate left (<= threshold) and right (> threshold)
5. Stop when reaching the desired number of leaves or no improving splits remain
