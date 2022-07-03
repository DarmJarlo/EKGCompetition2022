# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
import hrvanalysis as hrv
from hrvanalysis.preprocessing import remove_outliers, interpolate_nan_values
import random
import pandas as pd
from typing import List, Tuple
import pickle
import tsfel
import tensorflow as tf
import utils
from scipy import signal
from sklearn.decomposition import PCA


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden


def predict_labels(ecg_leads: List[np.ndarray], fs: float, ecg_names: List[str], model_name: str = 'model.npy',
                   is_binary_classifier: bool = False) -> List[Tuple[str, str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben,
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

    # ------------------------------------------------------------------------------
    # Euer Code ab hier
    with open(model_name, 'rb') as f:
        xgb = pickle.load(f)  # load boosting classifier

    with open('pca.pkl', 'rb') as pickle_file:
        pca = pickle.load(pickle_file)  # load trained pca

    detectors = Detectors(fs)
    cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')
    model = tf.saved_model.load('Keras_models/new_model')  # load trained ResNet 50

    feature_vector = np.array([])  # create empty arrays for features and predictions later
    predictions = list()
    predictions_res_list = list()

    artificial_idx = []
    extra_count = []
    ecg_leads_extra = []
    for index in range(len(ecg_leads)):
        if len(ecg_leads[index]) < 9000:
            lowiter = 9000 // len(ecg_leads[index])
            print(lowiter)
            artificial_idx.append(index)
            for i in range(lowiter):
                print(ecg_leads[index].shape)
                ecg_leads[index] = np.append(ecg_leads[index], ecg_leads[index])
                print('dadadad', ecg_leads[index].shape)
                artificial_idx.append(index + i)
            ecg_leads[index] = ecg_leads[index][0:9000]
            print(len(ecg_leads[index]))

    for idx in range(len(ecg_leads)):
        ecg_lead = ecg_leads[idx]
        spectral_features = tsfel.time_series_features_extractor(cfg, ecg_lead, fs=fs)
        corr_features = tsfel.correlated_features(spectral_features)
        spectral_features.drop(corr_features, axis=1, inplace=True)
        spectral_features = spectral_features.to_numpy()

        lead = utils.leads_transfer(ecg_lead, shape=(1, 90, 100, 1))
        prediction_res, features_res = model(lead)
        prediction_res = prediction_res.numpy()
        features_res = features_res.numpy()

        features_res = features_res.reshape((32, 64))
        kernel = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        features_res = signal.convolve2d(features_res, kernel)[::3, ::3]
        features_res = features_res.reshape(1, -1)
        features_res = pca.transform(features_res)[0, :80]
        predictions_res_list.append(prediction_res)

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
        feature_vector = np.append(feature_vector, features_res)

    feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 137), 137))  # reshape fv
    col = np.arange(137)
    index = np.arange(len(feature_vector))
    df = pd.DataFrame(data=feature_vector, index=index, columns=col)
    df = df.drop(df.columns[24], axis=1)  # column has None-values

    df = df.replace([np.inf, -np.inf], np.nan)  # Replace invalid values
    column_means = df.mean()
    df = df.fillna(column_means)
    df = df.fillna(0)

    feature_vector = df.to_numpy()

    predicted_classes = xgb.predict(feature_vector)

    idx_labels = np.arange(len(predicted_classes))
    columns_labels = np.arange(1)
    df_labels = pd.DataFrame(data=predicted_classes, index=idx_labels, columns=columns_labels)

    df_labels = df_labels.replace(to_replace=[0, 1, 2, 3], value=['N', 'A', 'O', '~'])

    predicted_classes = df_labels.to_numpy()

    for i in range(len(predicted_classes)):  # create prediction tuple
        predictions.append((ecg_names[i], predicted_classes[i][0]))

    # ------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
