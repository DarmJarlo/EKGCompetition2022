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

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
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

#------------------------------------------------------------------------------
# Euer Code ab hier
    Model = Network()
    Model = Model.load_model("model")
    with open(model_name, 'rb') as f:  
        rf = pickle.load(f)         # load random forest

    detectors = Detectors(fs)

    feature_vector = np.array([])  # create empty arrays for features and predictions later
    predictions = list()

    for idx, ecg_lead in enumerate(ecg_leads):
        rr_intervals = detectors.hamilton_detector(ecg_lead)
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

    feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 32), 32))
    feature_vector[:, 24] = 0  # column 24 has None-values

    feature_names = ['mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
                     'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'total_power', 'vlf', 'lf',
                     'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'triangular_index', 'tinn', 'sd1', 'sd2', 'ratio_sd2_sd1',
                     'csi', 'cvi', 'Modified_csi', 'sampen']

    index = np.arange(len(feature_vector))
    df = pd.DataFrame(data=feature_vector, index=index, columns=feature_names)

    df = df.replace([np.inf, -np.inf], np.nan)  # Replace invalid values
    column_means = df.mean()
    df = df.fillna(column_means)
    df = df.fillna(0)

    feature_vector = df.to_numpy()

    predicted_classes = rf.predict(feature_vector)

    idx_labels = np.arange(len(predicted_classes))
    columns_labels = np.arange(1)
    df_labels = pd.DataFrame(data=predicted_classes, index=idx_labels, columns=columns_labels)

    if is_binary_classifier:
        df_labels = df_labels.replace(to_replace=[0, 1], value=['N', 'A'])
    else:
        df_labels = df_labels.replace(to_replace=[0, 1, 2, 3], value=['N', 'A', 'O', '~'])

    predicted_classes = df_labels.to_numpy()

    for i in range(len(predicted_classes)):     # create prediction tuple
        predictions.append((ecg_names[i], predicted_classes[i][0]))


#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
