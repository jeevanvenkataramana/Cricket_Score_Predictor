import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import pickle
import sys

orig_stdout = sys.stdout
f_redirect = open('out_regression.txt', 'w')
sys.stdout = f_redirect


def get_features_and_labels(fileName):
    df = pd.read_csv(fileName)
    df = df[['balls', 'wickets', 'ground_average', 'pp_balls_left', 'total_overs', 'runs']]
    df_features = df.drop('runs', axis=1)
    df_label = df[['runs']]
    return df_features.as_matrix(), df_label.as_matrix()


# Get the best hyper-parameters using GridSearchCV
def hyperparameter_selection(train_x, train_y, param_grid, method):
    train_y = train_y.reshape(len(train_y), )
    clf = GridSearchCV(method, param_grid)
    clf.fit(train_x, train_y)
    return clf


# Regress on Lasso to predict the Classes for the test set
def regress(trainX, trainY, testX, testY, clf):
    clf.fit(trainX, trainY)
    y_predict = clf.predict(testX)

    with open('lasso.pickle', 'wb') as f:
        pickle.dump(clf, f)

    return math.sqrt(mean_squared_error(y_pred=y_predict, y_true=testY))

'''
def pipeline(X_tr, Y_tr):
    X_train, X_validate, y_train, y_validate = train_test_split(X_tr, Y_tr, test_size=0.2)

    # Get the best hyperparameters on selected features
    param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    method = LogisticRegression(penalty='l2')
    clf = hyperparameter_selection(X_train, y_train, param_grid, method)
    print "Pipeline Best Hyperparameter ", clf.best_params_['C']
    c = clf.best_params_['C']

    # Run the regressor on the selected features and best hyperparameter
    lasso = LogisticRegression(C=c)
    rmse = regress(X_train, y_train, X_validate, y_validate, lasso)
    print "Pipeline RMSE ", rmse
'''

X_tr, y_tr = get_features_and_labels('/Users/sbk/Score Predictor 255/data/Train_1st_Innings.csv')

X_train, X_validate, y_train, y_validate = train_test_split(X_tr, y_tr, test_size=0.2)

# Get the best hyperparameters on selected features
param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
method = LogisticRegression(penalty='l2')
clf = hyperparameter_selection(X_train, y_train, param_grid, method)
print "Pipeline Best Hyperparameter ", clf.best_params_['C']
c = clf.best_params_['C']

# Run the regressor on the selected features and best hyperparameter
lasso = LogisticRegression(C=c)
rmse = regress(X_train, y_train, X_validate, y_validate, lasso)
print "Pipeline RMSE ", rmse


#pipeline(X_tr, y_tr)
sys.stdout = orig_stdout
f_redirect.close()
