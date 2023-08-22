# Cancer-Inhibitors
Predicting cancer-inhibitor molecules based on chemical fingerprints/
# Inhibitor Classifier

This repository contains a comprehensive analysis and modeling of an inhibitor dataset, aiming to predict certain inhibitors based on various models and ultimately combining them using ensemble methods.

## Overview

1. **Data Loading and Preprocessing**: 
    - The data is loaded from Kaggle's directory for train and test datasets.
    - Certain rows are dropped based on specific conditions from the training dataset.
    - The datasets are then split into feature matrices (`X_train` and `X_test`) and target vectors (`y_train` and `y_test`).
    - Any missing values in the data are filled with zeros and data types are set as 'category' for features.
    - Principal Component Analysis (PCA) is used to reduce the dimensions of the dataset.

2. **Modeling**: 
    - Several machine learning models are trained on the dataset:
      * Support Vector Machine (SVM)
      * Random Forest
      * Gradient Boosting
      * K-Nearest Neighbors (KNN)
      * Gaussian Naive Bayes
      * Logistic Regression
    - Hyperparameter tuning is performed for the models using GridSearchCV to get the best parameters for each model.
    
3. **Ensemble Learning**:
    - After individual models are trained, an ensemble method called `VotingClassifier` is used. It takes into account the predictions from all the trained models to make a final prediction using majority voting.

4. **Evaluation**:
    - The performance of the ensemble classifier is evaluated using metrics such as:
      * Accuracy
      * Precision
      * Recall
      * F1 score

## Dataset

The dataset consists of inhibitors and is loaded from the paths:
- Training dataset: `/kaggle/input/inhibitors/cdk2_train.csv`
- Testing dataset: `/kaggle/input/inhibitors/cdk2_test.csv`

## Usage

1. Ensure you have the required libraries installed:
    - `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`

2. Run the code to perform data preprocessing, train various models, and evaluate the ensemble classifier.

3. Observe the performance metrics printed at the end of the run to understand the effectiveness of the ensemble classifier.

## Contribution

If you have any suggestions or find any bugs, please create an issue or submit a pull request.

## License

This project is open-source and available to everyone under the MIT License.
