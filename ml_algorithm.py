"""
Implementing a machine learning algorithm with extracted features of data_analysis
"""

import numpy as np
#import data_analysis as analysis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from wettbewerb import load_references
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost

ecg_leads, ecg_labels, fs, ecg_names = load_references()


def time_domain(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=False):
    features_time_domain = analysis.feature_extraction_time_domain(ecg_leads, ecg_labels, fs, four_problem, nn_intervals)
    X = features_time_domain[:,:-1]
    y = features_time_domain[:,-1:]

    #Source: https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
    indices = np.where(np.isnan(X))  # returns an array of rows and column indices
    for row, col in zip(*indices):
        X[row, col] = np.mean(X[~np.isnan(X[:, col]), col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestRegressor(n_estimators=150)
    rf.fit(X_train, y_train)
    feature_names_td = ['mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
                    'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr']

    sort = rf.feature_importances_.argsort()
    feature_names_td = [x for _,x in sorted(zip(sort, feature_names_td))]
    rf_features = rf.feature_importances_[sort]
    plt.barh(feature_names_td, rf_features)
    plt.xlabel('Feature Importance')
    plt.show()


def frequency_domain(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=False):
    features_frequency_domain = analysis.feature_extraction_frequency_domain(ecg_leads, ecg_labels, fs, four_problem, nn_intervals)
    X = features_frequency_domain[:,:-1]
    y = features_frequency_domain[:,-1:]

    indices = np.where(np.isnan(X))  # returns an array of rows and column indices
    for row, col in zip(*indices):
        X[row, col] = np.mean(X[~np.isnan(X[:, col]), col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestRegressor(n_estimators=150)
    rf.fit(X_train, y_train)
    feature_names_fd = ['total_power', 'vlf', 'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu']

    sort = rf.feature_importances_.argsort()
    feature_names_fd = [x for _,x in sorted(zip(sort, feature_names_fd))]
    rf_features = rf.feature_importances_[sort]
    plt.barh(feature_names_fd, rf_features)
    plt.xlabel('Feature Importance')
    plt.show()


def features_of_both(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=False):
    features_time_domain = analysis.feature_extraction_time_domain(ecg_leads, ecg_labels, fs, four_problem,
                                                                   nn_intervals)
    X_td = features_time_domain[:, :-1]
    y = features_time_domain[:, -1:]

    features_frequency_domain = analysis.feature_extraction_frequency_domain(ecg_leads, ecg_labels, fs, four_problem,
                                                                             nn_intervals)
    X_fd = features_frequency_domain[:, :-1]
    X = np.concatenate((X_td, X_fd), axis=1)

    indices = np.where(np.isnan(X))  # returns an array of rows and column indices
    for row, col in zip(*indices):
        X[row, col] = np.mean(X[~np.isnan(X[:, col]), col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf, X_test, y_test


def feature_training():
    df = pd.read_csv('../datasets/two_average_filtered_synth_extended.csv')
    #df = pd.read_csv('features_synth.csv')
    df = df.to_numpy()
    X = df[:, :-1]
    y = df[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=1500, bootstrap=False, max_features=10, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf, X_test, y_test


def tuning():
    forest = RandomForestClassifier(n_jobs=-1)
    df = pd.read_csv('../datasets/two_average_filtered_synth_extended.csv')
    df = df.to_numpy()
    X = df[:, :-1]
    y = df[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #forest.fit(X_train, y_train)
    param_grid = [
        {'n_estimators': [1000, 1500], 'max_features': [8, 10, 12], 'bootstrap': [True, False]}
    ]
    grid_search_forest = GridSearchCV(forest, param_grid, cv=10, scoring='f1_macro')
    grid_search_forest.fit(X_train, y_train)
    print(grid_search_forest.best_estimator_)


def xgb():
    df = pd.read_csv('../datasets/two_average_filtered_synth_extended.csv')
    df = df.to_numpy()
    X = df[:, :-1]
    y = df[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)

    return model, X_test, y_test

#tuning()
trained_rf, X_test, y_test = feature_training() # Random Forest
y_pred = trained_rf.predict(X_test)

#trained_xgb, X_test, y_test = xgb()  # XGBoosting
#y_pred = trained_xgb.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Precision:', metrics.precision_score(y_test, y_pred, average='macro')) #)) #, average=None))
print('Recall:', metrics.recall_score(y_test, y_pred, average='macro')) #)) #, average=None))
print('F1:', metrics.f1_score(y_test, y_pred, average='macro')) #)) #, average=None))

#sort = trained_rf.feature_importances_.argsort()
#feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8',
#                 'Feature 9', 'Feature 10', 'Feature 11', 'Feature 12', 'Feature 13', 'Feature 14', 'Feature 15',
#                 'Feature 16', 'Feature 17', 'Feature 18', 'Feature 19', 'Feature 20', 'Feature 21', 'Feature 22',
#                 'Feature 23', 'Feature 24', 'Feature 25', 'Feature 26', 'Feature 27', 'Feature 28', 'Feature 29',
#                 'Feature 30', 'Feature 31', 'Feature 32']
#feature_names = [x for _, x in sorted(zip(sort, feature_names))]
#rf_features = trained_rf.feature_importances_[sort]
#plt.figure(figsize=(8,6))
#plt.barh(feature_names, rf_features)
#plt.xlabel('Feature Importance')
#plt.show()