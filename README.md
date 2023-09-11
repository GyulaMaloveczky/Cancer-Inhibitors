# Cancer-Inhibitors
Analysis and modeling of cancer-inhibitor molecules using chemical fingerprints.

## Overview

1. **Data Loading and Preprocessing**: 
    - Loaded the data from Kaggle's directory for train and test datasets.
    - Observed during EDA that the train dataset and the test dataset had differing target variable distributions.

    ![image](https://github.com/GyulaMaloveczky/Cancer-Inhibitors/assets/117769460/3c2c6ade-6cc6-404f-8469-940c08500469)

    ![image](https://github.com/GyulaMaloveczky/Cancer-Inhibitors/assets/117769460/f56dee42-e1d3-4a64-86fb-a3a182999efd)

    - Dropped rows to oversample the training data so the distribution of the target variable matches the test data's.
    - Split the datasets into feature matrices (`X_train` and `X_test`) and target vectors (`y_train` and `y_test`).
    - Filled missing values in the data with zeros and set data types as 'category' for features.
    - Utilized Multiple Correspondence Analysis (MCA) instead of PCA for dimensionality reduction. Decided the number of components based on the 1:10 rule and considered the amount of information loss.
    
    ![image](https://github.com/GyulaMaloveczky/Cancer-Inhibitors/assets/117769460/d51dc5ce-306a-439e-a95a-9ff1a1e9f459)


2. **Modeling**: 
    - Trained the following machine learning models:
      * Hyperparameter-tuned Support Vector Machine (SVM)
      * Hyperparameter-tuned K-Nearest Neighbors (KNN)
      * Hyperparameter-tuned Gradient Boosting (GBOOST)
      * Hyperparameter-tuned Random Forest (Forest)
      * Gaussian Naive Bayes (Bayesian inference)
      * Logistic Regression (logreg)
    - Conducted hyperparameter tuning using GridSearchCV for SVM, KNN, GBOOST, and Forest.

3. **Ensemble Learning**:
    - Used a hard voting classifier, `VotingClassifier`, to ensemble the predictions from the models.

4. **Evaluation**:
    - Evaluated the performance of the ensemble classifier using:
      * Accuracy
      * Precision
      * Recall
      * F1 score

## Dataset

Utilized inhibitor datasets:
- Training dataset path: `/kaggle/input/inhibitors/cdk2_train.csv`
- Testing dataset path: `/kaggle/input/inhibitors/cdk2_test.csv`

##Note about the dataset
The apparant overfit was due the train and test being from different distributions. I have validated the same pipeline by mixing the train and test data and resplitting them randomly. It performed significantly better (f1 score of 0.85-0.95). 
## Libraries Imported

```python
import numpy as np 
import pandas as pd 
from sklearn import svm
import prince
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

