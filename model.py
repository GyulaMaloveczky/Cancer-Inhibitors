


import numpy as np
import pandas as pd



train = pd.read_csv("/kaggle/input/inhibitors/cdk2_train.csv", header = None)
test = pd.read_csv("/kaggle/input/inhibitors/cdk2_test.csv", header = None)
train = train.drop(train[(train[0]== 1) & ((train.index%5 == 1) |(train.index%5 == 2)|(train.index%5 == 3) )].index)






y_train = np.array(train.iloc[:,0])

y_test = np.array(test.iloc[:,0])

colsindex= range(0, train.shape[1])

X_test = pd.DataFrame(test.iloc[:,1:], columns = None).fillna(0).astype('category')
X_train = pd.DataFrame(train.iloc[:,1:], columns = None).fillna(0).astype('category')
X_test = X_test.reindex(columns=X_train.columns)
X_test = X_test.astype(X_train.dtypes)

#X_test = X_test.applymap(lambda x : x +1).astype('category')
#X_train = X_train.applymap(lambda x : x+1).asytpe('category')






from sklearn.decomposition import PCA
# Step 4: Fit the MCA model
pca = PCA(200)

# Step 5: Obtain the transformed data
X_train = pca.fit_transform(X_train)
X_test =pca.transform(X_test)


#X_test = ca.transform(X_test)
import matplotlib.pyplot as plt
import seaborn as sns










from sklearn import svm
from sklearn.model_selection import GridSearchCV


def CreateSVM():
    
# Define the parameter grid for hyperparameter tuning
    param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['poly', 'rbf'],
    'gamma': [0.1, 1, 10, 'scale', 'auto']
    }
    import sklearn.svm
    # Create an SVM model
    svm_model = svm.SVC(probability=True)
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Create a new model with the best parameters
    SVM = svm.SVC(**best_params)
    
    
    # Train the model
    SVM.fit(X_train, y_train)
    
    # Make predictions
    print('SVM', best_score, best_params)
    return SVM


def RandomForest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Create a Random Forest Classifier
    random_forest = RandomForestClassifier()
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Create a new Random Forest Classifier with the best parameters
    best_random_forest = RandomForestClassifier(**best_params)
    best_random_forest.fit(X_train, y_train)
    print('forest', best_score, best_params)
    # Make predictions using the best Random Forest Classifier
    return best_random_forest
    



def gBoost():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Create a Gradient Boosting Classifier
    gbm = GradientBoostingClassifier()
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Create a new Gradient Boosting Classifier with the best parameters
    best_gbm = GradientBoostingClassifier(**best_params)
    best_gbm.fit(X_train, y_train)
    print('gboost', best_score, best_params)
    # Make predictions using the best Gradient Boosting Classifier
    
    return best_gbm

def KNN():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_neighbors': [3,4, 5,6, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
    }
    
    # Create a KNN Classifier
    knn = KNeighborsClassifier()
    
    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Create a new KNN Classifier with the best parameters
    best_knn = KNeighborsClassifier(**best_params)
    best_knn.fit(X_train, y_train)
    print('KNN', best_score, best_params)
    # Make predictions using the best KNN Classifier
    return best_knn


from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes classifier
naive_bayes = GaussianNB()

# Fit the classifier to the training data
naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)
from sklearn.metrics import accuracy_score
print('Bayes',accuracy_score(y_test, y_pred))
# Make predictions on the test data

from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression classifier
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Logreg',accuracy_score(y_test, y_pred))
# Make predictions on the test data

knn = KNN()
SVM = CreateSVM()
gboost = gBoost()
forest = RandomForest()

from sklearn.ensemble import VotingClassifier


#pre_ensamble
#pre_ensamble= VotingClassifier(estimators=[('logreg', logreg),('SVM', SVM)], voting = 'soft')

ensemble = VotingClassifier(
    estimators=[('knn', knn), ('SVM', SVM), ('gboost', gboost),('forest', forest), ('logreg', logreg), ('naive_bayes', naive_bayes)],
    voting='hard'  # Use majority voting
)
ensemble.fit(X_train, y_train)

# Make predictions on the test data using the ensemble classifier
y_pred = ensemble.predict(X_test)








from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print(set(y_pred))
asd = y_pred-y_test
