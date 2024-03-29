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
import config


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
    predictions_labels : list of tuples
        ecg_name und eure Diagnose
    '''

    # ------------------------------------------------------------------------------
    # Euer Code ab hier
    with open(model_name, 'rb') as f:
        xgb = pickle.load(f)  # load boosting classifier
        model = tf.saved_model.load('Keras_models/new_model')  # load resnet

    detectors = Detectors(fs)
    cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

    feature_vector = np.array([])  # create empty arrays for features and predictions_labels later
    predictions_labels = list()
    predictions_res_list = list()

    ecg_leads_std, extra_index = utils.relength_leads(ecg_leads)  # make leads have uniform length

    if config.Oned != True:
        shape =(1,90,100,1)
    else:
        shape = (1,9000,1)

    for idx in range(len(ecg_leads_std)):
        print(idx)
        ecg_lead = ecg_leads_std[idx]
        spectral_features = tsfel.time_series_features_extractor(cfg, ecg_lead, fs=fs)
        corr_features = tsfel.correlated_features(spectral_features)
        spectral_features.drop(corr_features, axis=1, inplace=True)
        spectral_features = spectral_features.to_numpy()  # extract spectral features

        res_data = utils.leads_transfer(ecg_lead, shape=shape)
        feature1, feature2, feature3, feature4, prediction, feature4_p = utils.res_feature(res_data, model)
        prediction_res = prediction.numpy()
        predictions_res_list.append(prediction_res)  # get resnet prediction

        rr_intervals = detectors.two_average_detector(ecg_lead)  # get rr-intervals
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

        dict_time_domain = hrv.get_time_domain_features(rr_intervals_list) # hrv-features
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

    feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 57), 57))  # reshape fv
    col = np.arange(57)
    index = np.arange(len(feature_vector))
    df = pd.DataFrame(data=feature_vector, index=index, columns=col)
    df = df.drop(df.columns[24], axis=1)  # column has None-values

    df = df.replace([np.inf, -np.inf], np.nan)  # Replace invalid values
    column_means = df.mean()
    df = df.fillna(column_means)
    df = df.fillna(0)

    feature_vector = df.to_numpy()

    predicted_labels_xgb = xgb.predict_proba(feature_vector)  # combine predictions
    predictions_res = np.vstack(predictions_res_list)
    temp = 0.5*predicted_labels_xgb + 0.5*predictions_res
    y_pred = []
    for i in range(len(temp)):
        label = np.argmax(temp[i])
        y_pred.append(label)
    predicted_classes = np.vstack(y_pred)

    labels = []
    count = 0
    temp_arr = []
    for i in range(len(ecg_leads)):  # combine predictions for resized leads
        if len(extra_index) > 0:
            if count < len(extra_index):
                temp_arr = extra_index[count]
            if i == temp_arr[0]:
                temp = temp_arr
                temp_pred = []
                for j in range(len(temp)):
                    pred = predicted_classes[j]
                    temp_pred.append(pred)
                if 1 in temp_pred:
                    labels.append(1)
                    count += 1
                    continue
                if 3 in temp_pred:
                    labels.append(3)
                    count += 1
                    continue
                if 0 in temp_pred:
                    labels.append(0)
                    count += 1
                    continue
                if 2 in temp_pred:
                    labels.append(2)
                    count += 1
                    continue
            else:
                labels.append(predicted_classes[i])
        else:
            labels.append(predicted_classes[i])

    labels = np.array(labels)
    idx_labels = np.arange(len(labels))
    columns_labels = np.arange(1)
    df_labels = pd.DataFrame(data=labels, index=idx_labels, columns=columns_labels)

    df_labels = df_labels.replace(to_replace=[0, 1, 2, 3], value=['N', 'A', 'O', '~'])

    predicted_classes = df_labels.to_numpy()

    for i in range(len(predicted_classes)):  # create prediction tuple
        predictions_labels.append((ecg_names[i], predicted_classes[i][0]))

    # ------------------------------------------------------------------------------
    return predictions_labels  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!