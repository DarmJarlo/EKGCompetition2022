""" File for evaluating the combination of both models"""

import numpy as np
from wettbewerb import load_references
from scipy.signal import medfilt
from ecgdetectors import Detectors
import hrvanalysis as hrv
from hrvanalysis.preprocessing import get_nn_intervals, remove_outliers, interpolate_nan_values
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from data_analysis import uniform_length, smote_algo
import pickle
from Model_apply import res_feature, leads_transfer
import os
from res_train import train_res
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import tensorflow as tf


def uniform_dataframe(save=False):
    ecg_leads, ecg_labels, fs, ecg_names = load_references()
    df_uni = uniform_length(ecg_leads, ecg_labels)
    X = df_uni[:, :-1]
    y = df_uni[:, -1]

    X, y = smote_algo(X, y)
    col = np.arange(9000)
    index = np.arange(len(X))
    df = pd.DataFrame(data=X, index=index, columns=col)
    df = df.assign(Labels=y)

    if save:
        df.to_csv('../datasets/df_uniform.csv', encoding='utf-8', index=False)
    return df


def feature_extraction(ecg_leads, fs, save=False):
    detectors = Detectors(fs)

    feature_vector = np.array([])
    cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

    for i in range(len(ecg_leads)):
        ecg_lead = ecg_leads[i]
        spectral_features = tsfel.time_series_features_extractor(cfg, ecg_lead, fs=fs)
        corr_features = tsfel.correlated_features(spectral_features)
        spectral_features.drop(corr_features, axis=1, inplace=True)
        spectral_features = spectral_features.to_numpy()

        #rr_intervals = detectors.hamilton_detector(ecg_lead)
        rr_intervals = detectors.two_average_detector(ecg_lead)

        if len(rr_intervals) == 1:
            rr_intervals = np.abs(rr_intervals)
            arti_rr_1 = rr_intervals * random.random()
            rr_intervals = np.append(rr_intervals, [arti_rr_1])

        rr_intervals_ms = np.diff(rr_intervals) / fs * 1000  # Umwandlung in ms
        rr_intervals_ms = [abs(number) for number in rr_intervals_ms]

        rr_without_outliers = remove_outliers(rr_intervals_ms, low_rri=300, high_rri=2000)
        rr_intervals_list = interpolate_nan_values(rr_without_outliers, interpolation_method='linear')

        rr_intervals_list = [x for x in rr_intervals_list if str(x) != 'nan']  # remove nan values

        if len(rr_intervals_list) <= 2:
            mean_rr = np.nanmean(rr_intervals_list)
            rr_intervals_list = np.nan_to_num(rr_intervals, nan=mean_rr)
            arti_rr_1 = rr_intervals_list[0] * random.random()
            arti_rr_2 = rr_intervals_list[0] * random.random()
            rr_intervals_list = np.append(rr_intervals_list, arti_rr_1)
            rr_intervals_list = np.append(rr_intervals_list, arti_rr_2)

        rr_intervals_list = [abs(number) for number in rr_intervals_list]

        dict_time_domain = hrv.get_time_domain_features(rr_intervals_list)
        dict_geometrical_features = hrv.get_geometrical_features(rr_intervals_list)
        dict_pointcare = hrv.get_poincare_plot_features(rr_intervals_list)
        dict_csi_csv = hrv.get_csi_cvi_features(rr_intervals_list)
        dict_entropy = hrv.get_sampen(rr_intervals_list)
        dict_frequency_domain = hrv.get_frequency_domain_features(rr_intervals_list)


        values_time = list(dict_time_domain.values())
        values_frequency = list(dict_frequency_domain.values())
        values_geometrical = list(dict_geometrical_features.values())
        values_pointcare = list(dict_pointcare.values())
        values_entropy = list(dict_entropy.values())
        values_csicsv = list(dict_csi_csv.values())

        feature_vector = np.append(feature_vector, values_time)
        feature_vector = np.append(feature_vector, values_frequency)
        feature_vector = np.append(feature_vector, values_geometrical)
        feature_vector = np.append(feature_vector, values_pointcare)
        feature_vector = np.append(feature_vector, values_entropy)
        feature_vector = np.append(feature_vector, values_csicsv)
        feature_vector = np.append(feature_vector, spectral_features)

        if (i % 100) == 0:
               print(str(i) + "\t EKG Signale wurden verarbeitet.")

    feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 57), 57))

    feature_names = ['mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
                     'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'total_power', 'vlf', 'lf',
                     'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'triangular_index', 'tinn', 'sd1', 'sd2', 'ratio_sd2_sd1',
                     'csi', 'cvi', 'Modified_csi', 'sampen', 'Fundamental freq', 'Human energy range',
                     'Max power spectrum', 'Max Frequency', 'Median Frequency', 'Power bandwith',
                     'Spectral centroid', 'Spectral decrease', 'Spectral entropy', 'Spectral kurtosis',
                     'Spectral positive turning points', 'Spectral roll-off', 'Spectral roll-on', 'Spectral skewness',
                     'Spectral spread', 'Spectral variation', 'Wavelet abs mean 1', 'Wavelet abs mean 2',
                     'Wavelet abs mean 3', 'Wavelet abs mean 4', 'Wavelet abs mean 5', 'Wavelet abs mean 6',
                     'Wavelet abs mean 7', 'Wavelet abs mean 8', 'Wavelet abs mean 9']
    index = np.arange(len(feature_vector))
    df = pd.DataFrame(data=feature_vector, index=index, columns=feature_names)
    df = df.drop(columns=['tinn'])

    df = df.replace([np.inf, -np.inf], np.nan)
    column_means = df.mean()
    df = df.fillna(column_means)
    if save:
        df.to_csv('../datasets/features_uniform_length.csv', encoding='utf-8', index=False)
    feature_vector = df.to_numpy()

    return feature_vector, df


def features_res(data):
    f_list = []
    pred_list = []
    model = tf.saved_model.load('Keras_models/new_model')
    for i in range(len(data)):
        print(i)
        lead = data[i]
        leads = leads_transfer(lead,shape=(1,50,180,1))
        predictions, features = res_feature(model, leads)
        features = features.reshape(-1, 2048)
        #pca = PCA()
        #pca.fit(features)
        #print(np.round(pca.explained_variance_ratio_, 5)[:10].sum())
        #features = pca.transform(features)[:, :10]
        features = features.reshape(-1)
        f_list.append(features)
        pred_list.append(predictions)
    f_list = np.vstack(f_list)
    pred_list = np.vstack(pred_list)
    pca = PCA()
    pca.fit(f_list)
    print(np.round(pca.explained_variance_ratio_, 40)[:40].sum())
    f_list = pca.transform(f_list)[:, :40]
    return f_list, pred_list


def train_both(save=False):
    print('loading dataset')
    df = pd.read_csv('../datasets/df_uniform.csv')
    print('done')
    df = df.to_numpy()
    X = df[:, :-1]
    y = df[:, -1]
    #X = X[:1000]
    #y = y[:1000]
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Label_set_train = np.zeros((len(y_train), 4))
    Label_set_test = np.zeros((len(y_test), 4))

    for i in range(len(y_train)):
        print('111111',i,y_train[i])
        dummy_test = np.zeros(4)
        dummy_test[int(y_train[i])] = 1
        Label_set_train[i, :] = dummy_test

    for j in range(len(y_test)):
        print('111111',j,y_train[j])
        dummy_test = np.zeros(4)
        dummy_test[int(y_train[j])] = 1
        Label_set_test[j, :] = dummy_test

    train_res(X_train, Label_set_train)
    #X_train, y_train = uniform_length(X_train, y_train)
    features_resnet_train, pred_resnet_train = features_res(X_train)
    features_resnet_test, pred_resnet_test = features_res(X_test)

    features_xgb, df_xgb = feature_extraction(X_train, fs=300)
    #X_train_xgb = features_xgb
    X_train_xgb = np.concatenate((features_xgb, features_resnet_train), axis=1)
    y_train_xgb = np.array([y_train])
    y_train_xgb = y_train_xgb.T
    xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6, min_child_weight=0, gamma=0.1,
                          subsample=0.55, colsample_bytree=0.75, bjective='multi:softmax',
                          nthread=4, scale_pos_weight=1, seed=42)
    xgb.fit(X_train_xgb, y_train_xgb.T)
    if save:
        if os.path.exists("xgb_both_features.npy"):
            os.remove("xgb_both_features.npy")
        with open('xgb_both_features.npy', 'wb') as f:
            pickle.dump(xgb, f)  # save model

    print('Training is done')

    return xgb, features_resnet_test, pred_resnet_test, X_test, y_test


def test_both(xgb, res_features, pred_resnet, X_test, y_test, both=False):
    xgb_features, data = feature_extraction(X_test, fs=300)
    if both:
        pred_xgb = xgb.predict_proba(xgb_features)
        xgb_features = np.concatenate((xgb_features, res_features), axis=1)
        xgb_pred = xgb.predict(xgb_features)
        y_pred = np.array([])
        for i in range(len(xgb_pred)):
            label = np.argmax(xgb_pred[i])
            y_pred = np.append(y_pred, label)
        #f_list, pred_list = features_res(X_test)
        #for i in range(len(X_test)):
        #    print(i)
        #    lead = X_test[i]
        #    leads = leads_transfer(lead, shape=(1, 50, 180, 1))
        #    features, pred_res = features_res(leads)
        #    pred_list.append(pred_res)

        #pred_list = np.vstack(pred_list)
        #temp = pred_xgb + pred_resnet
        #y_pred = []
        #for i in range(len(temp)):
        #    label = np.argmax(temp[i])
        #    y_pred.append(label)
        #y_pred = np.vstack(y_pred)
    else:
        xgb_features = np.concatenate((xgb_features, res_features), axis=1)
        xgb_pred = xgb.predict(xgb_features)
        y_pred = np.array([])
        for i in range(len(xgb_pred)):
            label = np.argmax(xgb_pred[i])
            y_pred = np.append(y_pred, label)
        #y_pred = np.vstack(y_pred)
        #y_pred = y_pred.T
    print(y_pred)
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('Precision:', metrics.precision_score(y_test, y_pred, average=None))
    print('Recall:', metrics.recall_score(y_test, y_pred, average=None))
    print('F1:', metrics.f1_score(y_test, y_pred, average=None))


#xgb, features_resnet, pred_resnet, X_test, y_test = train_both()
#test_both(xgb, features_resnet, pred_resnet, X_test, y_test, both=True)
print('Loading dataset')
df = pd.read_csv('../datasets/df_uniform.csv')
print('done')
df = df.to_numpy()
X = df[:,:-1]
y = df[:,-1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#features_res(X_train)

#X_test = X_test[:50]
#y_test = y_test[:50]
#print('done')

#with open('model_4p_2.npy', 'rb') as f:
#    xgb = pickle.load(f)

#test_both(xgb, X_test, y_test)
