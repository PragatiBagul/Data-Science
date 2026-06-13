# Decision Tree for Mushroom Classification

## Overview

This project implements a **Decision Tree Classifier from scratch** to classify mushrooms as either edible or poisonous using the Mushroom dataset. The notebook also includes a comparison with a **Random Forest Classifier** from Scikit-learn.

The main objective is to demonstrate the inner workings of decision trees, including:

* Entropy calculation
* Information Gain
* Recursive tree construction
* Tree traversal for prediction
* Comparison with an industry-standard machine learning model

---

## Dataset

The project uses a dataset named:

```text
mushrooms.csv
```

The dataset contains categorical features describing various mushroom characteristics such as:

* Cap shape
* Cap color
* Odor
* Gill size
* Gill color
* Stalk properties
* Habitat
* And other biological attributes

### Target Variable

```text
class
```

Typical values:

* `e` → Edible
* `p` → Poisonous

---

## Project Workflow

### 1. Import Libraries

The notebook starts by importing:

```python
import pandas as pd
import numpy as np
```

These libraries are used for:

* Data manipulation
* Numerical calculations
* Entropy and Information Gain computations

---

### 2. Load Dataset

```python
data = pd.read_csv('./mushrooms.csv')
```

The mushroom dataset is loaded into a Pandas DataFrame.

---

### 3. Initial Data Exploration

Several exploratory commands are executed:

```python
data.head()
data.columns
len(data.nunique())
data.shape
data.info()
```

These help understand:

* Dataset dimensions
* Feature names
* Number of unique values
* Data types
* Missing value status

---

### 4. Train-Test Split

The notebook manually creates a training and testing dataset:

```python
df = data.sample(frac=0.8, random_state=42)
df_test = data.drop(df.index)
```

Result:

* 80% Training Data
* 20% Testing Data

---

## Custom Decision Tree Implementation

The notebook builds a decision tree entirely from scratch.

---

### Entropy Function

Entropy measures the impurity of a dataset.

```python
def entropy(s):
    probs = s.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))
```

### Formula

Entropy is calculated as:

H(S)=-\sum_i p_i\log_2(p_i)

Where:

* (p_i) is the probability of class (i)
* Lower entropy indicates purer data
* Higher entropy indicates more uncertainty

---

### Information Gain

Information Gain determines which feature provides the best split.

```python
def information_gain(df, split_attribute_name, target_attribute_name):
```

Steps:

1. Calculate total entropy.
2. Split data based on a feature.
3. Compute entropy for each subset.
4. Calculate weighted entropy.
5. Subtract weighted entropy from total entropy.

### Formula

IG(S,A)=H(S)-\sum_v \frac{|S_v|}{|S|}H(S_v)

Where:

* (H(S)) = original entropy
* (S_v) = subset after splitting
* (IG) = information gain

The feature with the highest information gain is selected for splitting.

---

## Tree Node Structure

A custom Node class is defined:

```python
class Node:
```

Each node stores:

| Attribute | Description                        |
| --------- | ---------------------------------- |
| feature   | Feature used for splitting         |
| value     | Parent branch value                |
| results   | Final class prediction (leaf node) |
| branches  | Child nodes                        |

---

## Recursive Tree Construction

The tree is built using:

```python
build_tree()
```

### Algorithm

1. Check if all samples belong to one class.
2. If yes, create a leaf node.
3. If no features remain, return the majority class.
4. Compute Information Gain for all remaining features.
5. Select the best feature.
6. Split the dataset.
7. Recursively build child nodes.

This follows the same fundamental idea used in the ID3 decision tree algorithm.

---

## Prediction Function

Predictions are generated using:

```python
predict(node, sample)
```

The function:

1. Starts at the root node.
2. Reads the relevant feature value.
3. Follows the matching branch.
4. Continues recursively.
5. Returns the class label when a leaf node is reached.

---

## Model Evaluation

Predictions are generated for the test set:

```python
pred = predict(tree, row)
```

The notebook compares:

```python
Predicted : ...
Output : ...
```

and counts classification errors.

---

## Random Forest Implementation

To compare performance, the notebook also trains a Random Forest model using Scikit-learn.

### Imports

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

---

### Feature Encoding

Because Random Forest requires numerical inputs:

```python
X = pd.get_dummies(X, drop_first=True)
```

One-hot encoding converts categorical variables into binary features.

---

### Train-Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### Model Training

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

Parameters:

* `n_estimators=100`

  * Number of trees in the forest
* `max_depth=10`

  * Maximum depth of each tree
* `random_state=42`

  * Reproducibility

---

### Prediction and Accuracy

```python
predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

The notebook prints:

```python
Model Accuracy ...
```

This provides an overall performance metric for the Random Forest classifier.

---

## Key Concepts Demonstrated

### Machine Learning

* Classification
* Supervised Learning
* Model Evaluation

### Decision Trees

* Entropy
* Information Gain
* Recursive Tree Building
* Leaf Nodes
* Feature Selection

### Random Forest

* Ensemble Learning
* Multiple Decision Trees
* Improved Generalization
* Reduced Overfitting

---

## Project Structure

```text
Decision Tree for Mushroom Classification.ipynb
mushrooms.csv
README.md
```

---

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

## Running the Notebook

1. Place `mushrooms.csv` in the project directory.
2. Open the notebook:

```bash
jupyter notebook
```

3. Run all cells sequentially.
4. Observe:

   * Dataset exploration
   * Decision Tree construction
   * Predictions
   * Random Forest accuracy

---

## Learning Outcomes

After completing this notebook, you will understand:

* How entropy measures uncertainty
* How information gain chooses optimal splits
* How a decision tree is built recursively
* How predictions are generated from a tree structure
* Why Random Forests often outperform single decision trees
* How to implement a basic machine learning algorithm without relying entirely on external libraries

---

## Future Improvements

Potential enhancements include:

* Tree visualization
* Pruning techniques
* Cross-validation
* Precision, Recall, and F1-score evaluation
* Confusion matrix visualization
* Feature importance analysis
* Comparison with other classifiers (SVM, Logistic Regression, XGBoost)

---

## Author Notes

This project is intended as an educational implementation of Decision Trees. The custom implementation helps demonstrate the core algorithmic concepts, while the Random Forest model provides a practical benchmark using a production-grade machine learning library.
