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
from scipy import signal
from Denoise import relength


def uniform_data():
    ecg_leads, ecg_labels, fs, ecg_names = load_references()
    ecg_leads_std, Label_set, extra_index = relength(ecg_leads, ecg_labels)
    print(extra_index)

#uniform_data()

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
        df.to_csv('../datasets/df_uniform_2.csv', encoding='utf-8', index=False)
    return df
#uniform_dataframe(save=False)

def combine_df():
    df_res_features = pd.read_csv('../datasets/res_features.csv')
    X_res = df_res_features.to_numpy()
    df_xgb_features = pd.read_csv('../datasets/features_uniform_length.csv')
    df_xgb_features = df_xgb_features.to_numpy()
    X_xgb = df_xgb_features[:, :-1]
    y_xgb = df_xgb_features[:, -1]
    features = np.concatenate((X_xgb, X_res), axis=1)
    col = np.arange(X_xgb.shape[1] + X_res.shape[1])
    index = np.arange(len(X_xgb))
    df = pd.DataFrame(data=features, index=index,columns=col)
    df = df.assign(Labels=y_xgb)
    df.to_csv('../datasets/features_combined.csv', encoding='utf-8', index=False)
    print('done')

#combine_df()

def feature_extraction(save=False):
    ecg_leads, ecg_labels, fs, ecg_names = load_references()
    ecg_leads_std, Label_set, extra_index = relength(ecg_leads, ecg_labels)

    detectors = Detectors(fs)

    feature_vector = np.array([])
    cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

    ecg_labels_std = []
    ecg_leads_extra = []
    ecg_labels_extra = []
    extra_index = []

    n=len(ecg_leads)
    index_plus = n - 1
    for index in range(len(ecg_labels)):
        if ecg_labels[index] == 'N':
            ecg_labels_std.append(0)
        elif ecg_labels[index] == 'A':
            ecg_labels_std.append(1)
        elif ecg_labels[index] == 'O':
            ecg_labels_std.append(2)
        elif ecg_labels[index] == '~':
            ecg_labels_std.append(3)

        if len(ecg_leads[index]) < 9000:
            lowiter = 9000 // len(ecg_leads[index])
            print(lowiter)
            for i in range(lowiter):
                print(ecg_leads[index].shape)
                ecg_leads[index] = np.append(ecg_leads[index], ecg_leads[index])
                print('dadadad', ecg_leads[index].shape)
            ecg_leads[index] = ecg_leads[index][0:9000]
            print(len(ecg_leads[index]))
        elif len(ecg_leads[index]) > 9000:
            extra_index_block=[]
            if len(ecg_leads[index] <= 18000):
                ecg_leads[index] = ecg_leads[index][0:9000]
                extra_index_block.append(index)
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                index_plus = index_plus+1
                extra_index_block.append(index_plus)
                ecg_labels_extra.append(ecg_labels_std[index])
            elif len(ecg_leads[index] > 18000):
                iter = len(ecg_leads[index]) // 9000
                ecg_leads[index] = ecg_leads[index][:9000]
                extra_index_block.append(index)
                index_plus = index_plus+1
                for i in range(1, iter):
                    start = 9000 * i
                    end = 9000 * (i + 1)
                    index_plus=index_plus+1
                    extra_index_block.append(index_plus)
                    ecg_leads_extra.append(ecg_leads[start:end])
                    ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                index_plus = index_plus+1
                extra_index_block.append(index_plus)

                ecg_labels_extra.append(ecg_labels_std[index])
            extra_index.append(extra_index_block)
    ecg_labels_std = ecg_labels_std + ecg_labels_extra
    ecg_leads_std = ecg_leads + ecg_leads_extra

    for i in range(len(ecg_leads_std)):
        ecg_lead = ecg_leads_std[i]
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
    df = df.assign(Labels=ecg_labels_std)
    if save:
        df.to_csv('../datasets/xgb_uni_length_without_SMOTE.csv', encoding='utf-8', index=False)
    feature_vector = df.to_numpy()

    return feature_vector, df

feature_extraction(save=True)

def features_res(data, save=False):
    f_list = []
    pred_list = []
    model = tf.saved_model.load('Keras_models/new_model')
    for i in range(len(data)):
        print(i)
        lead = data[i]
        leads = leads_transfer(lead,shape=(1,90,100,1))
        predictions, features = res_feature(model, leads)
        features = features.reshape((32, 64))
        kernel = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        features = signal.convolve2d(features, kernel)[::3, ::3]
        features = features.reshape(-1)
        f_list.append(features)
        pred_list.append(predictions)
    f_list = np.vstack(f_list)
    pred_list = np.vstack(pred_list)
    pca = PCA()
    pca.fit(f_list)
    acc = np.round(pca.explained_variance_ratio_, 80)[:80].sum()
    print(acc)
    f_list = pca.transform(f_list)[:, :80]  # 72 for 90%

    with open('pca.pkl', 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)
    if save:
        columns = np.arange(80)
        index = np.arange(len(f_list))
        df = pd.DataFrame(data=f_list, index=index, columns=columns)
        df.to_csv('../datasets/res_features.csv', encoding='utf-8', index=False)
    return f_list, pred_list


def train_both(save=False):
    print('loading dataset')
    df_samples = pd.read_csv('../datasets/df_uniform.csv')
    print('done')
    df_res = df_samples.to_numpy()
    X_res = df_res[:, :-1]
    y_res = df_res[:, -1]
    #X_res = X_res[:1000]
    #y_res = y_res[:1000]
    y_res = LabelEncoder().fit_transform(y_res)
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    Label_set_train = np.zeros((len(y_train_res), 4))
    Label_set_test = np.zeros((len(y_test_res), 4))

    for i in range(len(y_train_res)):
        print('111111',i,y_train_res[i])
        dummy_test = np.zeros(4)
        dummy_test[int(y_train_res[i])] = 1
        Label_set_train[i, :] = dummy_test

    for j in range(len(y_test_res)):
        print('111111',j,y_test_res[j])
        dummy_test = np.zeros(4)
        dummy_test[int(y_test_res[j])] = 1
        Label_set_test[j, :] = dummy_test

    train_res(X_train_res, Label_set_train)
    features_resnet_train, pred_resnet_train = features_res(X_train_res)
    features_resnet_test, pred_resnet_test = features_res(X_test_res)

    #print('loading dataset')
    #df_features = pd.read_csv('../datasets/features_uniform_length.csv')
    #print('done')
    #df_features = df_features.to_numpy()
    #X_xgb = df_features[:, :-1]
    #y_xgb = df_features[:, -1]
    #X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)
    X_train_xgb = X_train_res
    X_test_xgb = X_test_res
    features_xgb, df_xgb = feature_extraction(X_train_xgb, fs=300)
    X_train_xgb = features_xgb
    y_train_xgb = y_train_res
    y_test_xgb = y_test_res
    X_train_xgb = np.concatenate((X_train_xgb, features_resnet_train), axis=1)
    y_train_xgb = np.array([y_train_xgb])
    y_train_xgb = y_train_xgb.T
    xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=10, min_child_weight=0, gamma=0.1,
                          subsample=0.55, colsample_bytree=0.75, bjective='multi:softmax',
                          nthread=-1, scale_pos_weight=1, seed=42)
    xgb.fit(X_train_xgb, y_train_xgb.T)
    if save:
        if os.path.exists("xgb_both_features.npy"):
            os.remove("xgb_both_features.npy")
        with open('xgb_both_features.npy', 'wb') as f:
            pickle.dump(xgb, f)  # save model

    print('Training is done')

    return xgb, features_resnet_test, pred_resnet_test, X_test_xgb, y_test_xgb


def test_both(xgb, res_features, pred_resnet, X_test_xgb, y_test_xgb, both=False):
    xgb_features, data = feature_extraction(X_test_xgb, fs=300)
    X_test_xgb = xgb_features

    if both:
        pred_xgb = xgb.predict_proba(xgb_features)
        xgb_features = np.concatenate((xgb_features, res_features), axis=1)
        xgb_pred = xgb.predict(xgb_features)
        y_pred = np.array([])
        for i in range(len(xgb_pred)):
            label = np.argmax(xgb_pred[i])
            y_pred = np.append(y_pred, label)
        f_list, pred_list = features_res(X_test)
        for i in range(len(X_test)):
            print(i)
            lead = X_test[i]
            leads = leads_transfer(lead, shape=(1, 50, 180, 1))
            features, pred_res = features_res(leads)
            pred_list.append(pred_res)

        pred_list = np.vstack(pred_list)
        temp = pred_xgb + pred_resnet
        y_pred = []
        for i in range(len(temp)):
            label = np.argmax(temp[i])
            y_pred.append(label)
        y_pred = np.vstack(y_pred)
    else:
        xgb_features = np.concatenate((X_test_xgb, res_features), axis=1)
        y_pred = xgb.predict(xgb_features)
        #y_pred = np.array([])
        #for i in range(len(xgb_pred)):
        #    label = np.argmax(xgb_pred[i])
        #    y_pred = np.append(y_pred, label)
        #y_pred = np.vstack(y_pred)
        #y_pred = y_pred.T
    print(y_pred)
    print('Accuracy:', metrics.accuracy_score(y_test_xgb, y_pred))
    print('Precision:', metrics.precision_score(y_test_xgb, y_pred, average=None))
    print('Recall:', metrics.recall_score(y_test_xgb, y_pred, average=None))
    print('F1:', metrics.f1_score(y_test_xgb, y_pred, average=None))


def train_test_res():
    print('loading dataset')
    df_samples = pd.read_csv('../datasets/df_uniform_2.csv')
    print('done')
    df_res = df_samples.to_numpy()
    X_res = df_res[:, :-1]
    y_res = df_res[:, -1]
    #X = X[:1000]
    #y = y[:1000]
    y_res = LabelEncoder().fit_transform(y_res)
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    Label_set_train = np.zeros((len(y_train_res), 4))
    Label_set_test = np.zeros((len(y_test_res), 4))

    for i in range(len(y_train_res)):
        print('111111', i, y_train_res[i])
        dummy_test = np.zeros(4)
        dummy_test[int(y_train_res[i])] = 1
        Label_set_train[i, :] = dummy_test

    for j in range(len(y_test_res)):
        print('111111', j, y_test_res[j])
        dummy_test = np.zeros(4)
        dummy_test[int(y_test_res[j])] = 1
        Label_set_test[j, :] = dummy_test

    train_res(X_train_res, Label_set_train)
    features_resnet_test, pred_resnet_test = features_res(X_test_res)
    y_pred = np.array([])
    for i in range(len(pred_resnet_test)):
        label = np.argmax(pred_resnet_test[i])
        y_pred = np.append(y_pred, label)

    print('Accuracy:', metrics.accuracy_score(y_test_res, y_pred))
    print('Precision:', metrics.precision_score(y_test_res, y_pred, average=None))
    print('Recall:', metrics.recall_score(y_test_res, y_pred, average=None))
    print('F1:', metrics.f1_score(y_test_res, y_pred, average=None))

#train_test_res()

#xgb, features_resnet, pred_resnet, X_test_xgb, y_test_xgb = train_both()
#test_both(xgb, features_resnet, pred_resnet, X_test_xgb, y_test_xgb, both=False)
#print('Loading dataset')
#df = pd.read_csv('../datasets/df_uniform_2.csv')
#print('done')
#print(df.empty)
#df = df.to_numpy()
#X = df[:,:-1]
#y = df[:,-1]
#features, df = feature_extraction(X, fs=300, save=True)
#features_res(X, save=True)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#features_res(X_train)

#X_test = X_test[:50]
#y_test = y_test[:50]
#print('done')

#with open('model_4p_2.npy', 'rb') as f:
#    xgb = pickle.load(f)

#test_both(xgb, X_test, y_test)
