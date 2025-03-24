# Diagnosing-Cardiovascular-Disease

 ## Table of Contents
- [Introduction](#introduction)
- [Required Libraries](#required-libraries)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [Gradient Boosting](#gradient-boosting)
- [Conclusion](#conclusion)
- [Credits](#credits)

## Introduction

This project implements & compares 3 tree-based machine learning models to predict the likelihood of cardiovascular disease. This enables earlier detection, so action can be taken to potentially save lives!

Those models are: decision tree, random forest, and gradient boosting.

They learn from health data on a large set of patients, and predict disease likelihood from the same health metrics on any new person.

## Required Libraries

To run the notebook in this repository, the libraries needed are:

- NumPy
- pandas
- scikit-learn
- XGBoost
- Matplotlib

Imported by doing:
```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
```

## Dataset Overview

The models are trained on the CSV file downloaded from Kaggle's [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). 

## Data Preprocessing

For categorical input features to be accepted by any machine learning algorithm, it is one-hot encoded to avoid implicit ordering, maintain categorical independence, and make it binary-friendly. We did this using the pandas library's `get_dummies` function.

## Model Implementation

### Decision Tree
Using scikit-learn, I implement a decision tree model with varying limits on the minimum samples required per node to split, and maximum tree depth to observe their impact on performance.

Observe that increasing the minimum limit of elements for splitting generalizes well and reduces overfitting, especially around 50 samples to split:

![image](https://github.com/user-attachments/assets/f8ee9cff-3836-4237-8f62-70b64376845c)

However, increasing maximum splitting depth too far makes it overfit, as observed by the divergence in training & test set accuracies of the model:

![image](https://github.com/user-attachments/assets/9861e1f5-37c0-4c8e-a21d-3594cef644bd)

Tuning hyperparameters to optimize generalization gives us minimum sample split to be 50, and maximum recursion depth to be 4. *However*, the accuracy over the test & train datsets isn't great, staying at around 86%.

### Random Forest

The Random Forest algorithm will train multiple decision trees, each of which is trained on a random subset of the training data, and chooses node splits from a random subset of data features. Generally, each tree will split from $$\sqrt{n}$$ out of n total features, a common choice for balancing good performance and independence of trees.

I then used scikit-learn again to implement it, by calling the library's `RandomForestClassifier` setup, in which I controlled the minimum samples per node to split, maximum depth, and number of trees in the forest. The choice of random subsets is done automatically by the classifier, I simply give it all the data.

Using Matplotlib to observe variations in accuracy based on the hyperparameters of max depth, min samples to split, and number of estimators, I can see that for number of estimators, the train & test accuracies proceed as below. Notice that the best value is not necessarily more trees, and for this dataset, it is around 100. 

![image](https://github.com/user-attachments/assets/a22f7a88-1217-4ae8-8d48-56b7a8c71f42)

Tuning it so `max_depth=16`, `min_samples_split=10`, and `n_estimators=100`, we have a train accuracy of 93% and a test accuracy of 88%. 

A next step would be to use Sklearn's GridSearchCV to refit better models. 

### Gradient Boosting

**The Algorithm**

The gradient boosting [algorithm](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) makes an additional improvement over Random Forest. It improves the strength of the model by making newer trees correct the error of past trees.

As with all supervised learning algorithms, it will optimize to minimize an objective function $$\mathcal{J}$$ of the prediction w.r.t. the target. The generic function with a typical logistic loss & regularization term is:

$$\mathcal{J} = \sum_{i=1}^{N} \mathcal{L}(y_i, \hat{y_i}) + \sum_{k=1}^{K} \Omega (f_k)$$

Given that the performance of trees are aggregated, in a technique called the additive strategy, where every *new* prediction at step t is a function of the previous aggregate prediction $$\hat{y_i}^{(t-1)}$$ and the new tree $$f_t(x_i)$$.

$$\mathcal{J}^{(t)} = \sum_{i=1}^{N} \mathcal{L}(y_i, \hat{y_i}^{(t-1)} + f_t(x_i)) + \sum_{k=1}^{K} \Omega (f_k)$$

So, for a nice & differentiable MSE loss: $$\mathcal{J}^{(t)} = \sum_{i=1}^{N} \left( 2 \left( \hat{y}_i^{(t-1)} - y_i \right) f_t(x_i) + f_t(x_i)^2 \right) + \omega(f_t) + \text{constant}$$

Taking the Taylor expansion to the second order gets: $$\mathcal{J}^{(t)} = \sum_{i=1}^{N} [g_i f_t(x_i) + \frac{1}{2} h_i {f_t}^2(x_i)] + \omega(f_t)$$

In the loss function, $$g_i$$ is the first order gradient and $$h_i$$ is the second order Hessian, of the gradient for each data point.

*Here's the catch*: each new tree is *fit* to the *negative gradient* of the current loss $$\mathcal{J}^{(t)}$$, just like regular gradient descent! Then, the entire model is aggregated using a learning rate to preserve some old structure while improving over loss.

New predictions are hence $$\hat{y_i}^{(t)} = \hat{y_i}^{(t-1)} + \eta \cdot f_t(x_i)$$ - where $$\eta$$ is learning rate.

For more specificity, a tree's function $$f(x)$$ can be formulated as $$f_t(x) = w_{q(x)}$$ where $$w$$ is a vector of scores on leaves with $$q$$ a function assigning a datapoint to a leaf. 

Fitting new trees to the loss will go as: $${w_j}^* = - \frac{G_j}{H_j + \lambda}$$ where $$G_j$$ and $$H_j$$ are sums of gradients & Hessians over all data points and $$\lambda$$ is a regularization hyperparameter.

**The Implementation**

Using the very powerful **XGBoost** library to implement the algorithm, we call `XGBClassifier` with controls over a parameter called `early_stopping_rounds`, the number of iterations to continue training since the last accuracy improvement. 

By fitting only 26 estimator trees, over 16 training rounds, we have obtained the best accuracy possible, which is 92.5% on training set, and 86.4% on test set. Clearly very efficient!

## Conclusion

Random Forest & Gradient Boosting had similar accuracy results with little overfitting for both, and Gradient Boosting being more sample & computationally efficient.

## Credits

This project was completed as part of Stanford's Advanced Learning Algorithms course.
