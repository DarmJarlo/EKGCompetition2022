"""
Implementing a machine learning algorithm with extracted features of data_analysis
"""

import numpy as np
import data_analysis as analysis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from wettbewerb import load_references

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


def feature_training(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=True):
    features, targets = analysis.feature_extraction(ecg_leads, ecg_labels, fs, four_problem, nn_intervals)

    X = features
    y = targets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf, X_test, y_test


trained_rf, X_test, y_test = feature_training(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=False)
y_pred = trained_rf.predict(X_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Precision:', metrics.precision_score(y_test, y_pred, average=None)) #)) #,average=None))
print('Recall:', metrics.recall_score(y_test, y_pred, average=None)) #)) #, average=None))
print('F1:', metrics.f1_score(y_test, y_pred, average=None)) #)) #, average=None))