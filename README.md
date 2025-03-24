﻿# Diagnosing-Cardiovascular-Disease

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
A Decision Tree model is implemented using scikit-learn with varying limits on the minimum samples required per node to split, and maximum tree depth to observe their impact on performance.

Observe that increasing the minimum limit of elements for splitting generalizes well and reduces overfitting:
![image](https://github.com/user-attachments/assets/f8ee9cff-3836-4237-8f62-70b64376845c)

However, increasing maximum splitting depth too far makes it overfit, as observed by the divergence in training & test set accuracies of the model
![image](https://github.com/user-attachments/assets/9861e1f5-37c0-4c8e-a21d-3594cef644bd)

Tuning hyperparameters to optimize generalization gives us minimum sample split to be 50, and maximum recursion depth to be 4. *However*, the overall accuracy isn't great, staying at around 86% accuracy.

### Random Forest

### Gradient Boosting

Implemented using XGBoost.

## Conclusion

## Credits

This project was completed as part of Stanford's Advanced Learning Algorithms [course](https://github.com/AliesTaha/Stanford-ML) 
