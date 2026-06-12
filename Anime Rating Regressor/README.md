# 🎌 Anime Rating Prediction

### Machine Learning Regression Project using Anime Metadata

---

## 📌 Overview

Anime Rating Prediction is a supervised machine learning project that predicts an anime's user rating based on its characteristics and metadata.

Instead of relying on user reviews or recommendation systems, this project focuses on predicting ratings using information available before a user watches the anime, such as:

* Genre
* Number of Episodes
* Anime Type (TV, Movie, OVA, ONA, Special)
* Popularity Metrics

The project serves as an excellent introduction to:

* Regression Problems
* Feature Engineering
* Handling Categorical Variables
* Missing Value Treatment
* Model Evaluation
* Feature Importance Analysis

---

# 🎯 Problem Statement

Can we accurately predict an anime's rating using only its metadata?

Given information such as:

```text
Anime Name
Genre
Type
Episodes
Members
Popularity
```

Predict:

```text
Anime Rating
```

This is a **Regression Problem** because the target variable is continuous.

Example:

| Anime           | Actual Rating |
| --------------- | ------------- |
| Attack on Titan | 8.9           |
| Death Note      | 8.7           |
| Naruto          | 7.9           |

The model attempts to predict these ratings automatically.

---

# 📊 Dataset

Dataset Used:

**Anime Recommendations Database**

The dataset contains information about thousands of anime titles, including:

| Feature  | Description                                   |
| -------- | --------------------------------------------- |
| anime_id | Unique anime identifier                       |
| name     | Anime title                                   |
| genre    | Anime genres                                  |
| type     | TV, Movie, OVA, ONA, Special                  |
| episodes | Number of episodes                            |
| rating   | Average user rating                           |
| members  | Number of users who interacted with the anime |

---

# 🔍 Project Objectives

The primary goals of this project are:

* Predict anime ratings accurately.
* Compare multiple regression algorithms.
* Understand the impact of different features.
* Learn preprocessing techniques for structured datasets.
* Analyze feature importance.

---

# 🏗️ Machine Learning Pipeline

## 1. Data Collection

Load the anime dataset into Pandas.

```python
import pandas as pd

df = pd.read_csv("anime.csv")
```

---

## 2. Exploratory Data Analysis (EDA)

Perform analysis on:

* Rating distribution
* Genre frequencies
* Episode counts
* Popularity trends
* Missing values

Example questions:

* Do longer anime receive higher ratings?
* Which genres are rated highest?
* Are movies rated differently than TV series?

---

## 3. Data Cleaning

Handle:

### Missing Ratings

```python
df.dropna()
```

or

```python
df.fillna()
```

### Missing Episode Counts

Convert unknown values appropriately.

Example:

```text
Episodes = "Unknown"
```

Convert to:

```text
NaN
```

Then impute or remove.

---

## 4. Feature Engineering

### Genre Encoding

Genres are multi-label categories:

```text
Action, Adventure, Fantasy
```

Convert using:

* MultiLabelBinarizer
* One-Hot Encoding

Example:

| Action | Fantasy | Comedy |
| ------ | ------- | ------ |
| 1      | 1       | 0      |

---

### Type Encoding

Convert:

```text
TV
Movie
OVA
ONA
Special
```

Using:

```python
OneHotEncoder
```

---

### Popularity Features

Use:

```text
Members
Favorites
Rank
Popularity
```

if available.

---

## 5. Feature Scaling

Required for some models:

```python
StandardScaler
```

or

```python
MinMaxScaler
```

---

## 6. Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

Example:

```python
X_train, X_test, y_train, y_test
```

Training:

```text
80%
```

Testing:

```text
20%
```

---

# 🤖 Models Used

## 1. Linear Regression

Baseline model.

### Advantages

* Simple
* Fast
* Interpretable

### Limitations

* Assumes linear relationships
* Struggles with complex interactions

---

## 2. Random Forest Regressor

Ensemble tree-based model.

### Advantages

* Handles nonlinear patterns
* Robust to outliers
* Requires minimal scaling

### Why It Often Performs Better

Anime ratings depend on complex feature interactions:

Example:

```text
Genre + Episodes + Popularity
```

Random Forest captures these interactions naturally.

---

## 3. XGBoost Regressor

Gradient boosting model.

### Advantages

* State-of-the-art performance
* Handles missing values
* Captures complex relationships
* Strong regularization

Often achieves the best prediction accuracy.

---

# 📈 Model Evaluation

Evaluation Metrics:

## Mean Absolute Error (MAE)

Measures average prediction error.

MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|

Lower is better.

---

## Mean Squared Error (MSE)

Penalizes large errors more heavily.

MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2

---

## Root Mean Squared Error (RMSE)

More interpretable because it is in rating units.

RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}

---

## R² Score

Measures variance explained by the model.

[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
]

Closer to 1 indicates better performance.

---

# 🔬 Feature Importance Analysis

One major objective of this project is understanding:

> What makes an anime highly rated?

Using Random Forest and XGBoost feature importance scores, we can identify:

* Most influential genres
* Importance of episode count
* Effect of popularity
* Impact of anime type

Example:

| Feature      | Importance |
| ------------ | ---------- |
| Members      | 34%        |
| Genre_Action | 21%        |
| Episodes     | 17%        |
| Type_TV      | 12%        |
| Genre_Comedy | 8%         |

---

# 📊 Visualizations

The project includes:

### Rating Distribution

Histogram of ratings.

### Genre Analysis

Average rating by genre.

### Correlation Heatmap

Relationships among features.

### Feature Importance Plot

Most influential predictors.

### Prediction vs Actual Plot

Evaluate model performance visually.

---

# 🧠 Skills Demonstrated

This project showcases:

* Data Cleaning
* Missing Value Handling
* Feature Engineering
* Multi-Label Encoding
* Regression Modeling
* Ensemble Learning
* Model Evaluation
* Feature Importance Analysis
* Exploratory Data Analysis

---

# 💼 Interview Discussion Topics

This project naturally leads to several common ML interview questions.

### Why is this a Regression Problem?

Because the target variable (rating) is continuous.

---

### Why might Random Forest outperform Linear Regression?

Because anime ratings are influenced by nonlinear interactions among features.

Linear Regression assumes:

[
y = mx + b
]

while Random Forest learns complex decision boundaries.

---

### How do you handle categorical features?

Methods include:

* One-Hot Encoding
* Label Encoding
* Target Encoding
* MultiLabelBinarizer (for genres)

---

### How do you handle missing values?

Options:

* Mean Imputation
* Median Imputation
* Mode Imputation
* Model-based Imputation
* Row Removal

Choice depends on the feature and amount of missing data.

---

### Why use XGBoost?

Because it:

* Handles nonlinear relationships
* Supports regularization
* Manages missing values efficiently
* Usually achieves strong predictive performance

---

# 🚀 Future Improvements

* Add Anime Synopsis NLP Features
* Use User Review Sentiment Analysis
* Build an Anime Recommendation Engine
* Deploy using Flask/FastAPI
* Create Interactive Dashboard with Streamlit
* Experiment with Deep Learning Models
* Incorporate User-Based Collaborative Filtering

---
