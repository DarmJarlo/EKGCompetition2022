"""
Datei ist für Analyse der Daten gedacht. Welche Sachen sind auffällig, wie können die Klassen unterschieden werden.
"""

import numpy as np
from wettbewerb import load_references
from collections import Counter
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


#ecg_leads, ecg_labels, fs, ecg_names = load_references()
#a = dict(Counter(ecg_labels))
#print(a)

"Normal: 3581, Other: 1713, Noise: 185, AFib: 521"


def standardize(ecg_leads, cropping=False):
    """Cropping the data"""
    if cropping:
        size = []
        for i in range(len(ecg_leads)):
            ecg_lead = ecg_leads[i]
            size.append(len(ecg_lead))
        smallest_length = min(size)  # smallest sample = 2714

        for i in range(len(ecg_leads)):
            ecg_leads[i] = ecg_leads[i][:smallest_length]

    "Standardize the data"
    means = []
    stds = []
    for i in range(len(ecg_leads)):
        mean = np.mean(ecg_leads[i])
        std = np.std(ecg_leads[i])
        means.append(mean)
        stds.append(std)

    data_mean = np.mean(means)
    data_std = np.mean(stds)

    ecg_leads_standardized = []
    for i in range(len(ecg_leads)):
        new_ecgs = (ecg_leads[i] - data_mean) / data_std
        ecg_leads_standardized.append(new_ecgs)
        if (i % 100)==0:
            print(str(i) + "\t EKG Signale wurden verarbeitet.")

    return ecg_leads_standardized


"""Removal of Baseline-Drift"""
def baseline(ecg_leads, fs):
    ecg_leads_baselined = []
    kernel_1 = 0.2/(1/fs)    # Wurden auf 200 und 600ms gesetzt nach dem Paper
    kernel_2 = 0.6/(1/fs)
    kernel_1 = int(kernel_1)
    kernel_2 = int(kernel_2)
    if (kernel_1 % 2) == 0:
        kernel_1 += 1           # Kernel müssen odd sein
    if (kernel_2 % 2) == 0:
        kernel_2 += 1

    for i in range(len(ecg_leads)):
        first = medfilt(ecg_leads[i], kernel_1)
        second = medfilt(first, kernel_2)
        second = second.astype(ecg_leads[i].dtype)
        ecg_leads[i] -= second
        ecg_leads_baselined.append(ecg_leads[i])
        if (i % 100)==0:
            print(str(i) + "\t EKG Signale wurden verarbeitet.")

    return ecg_leads_baselined


""" 
Feature Extraction: Time and Frequency Domain possible
- Frequency domain aber eigentlich nur bei Samples zwischen 2-5 Minuten
- Features werden pro Iteration in einem Dictionary zurückgegeben und in einer Liste gespeichert
- Flags geben an, ob 2-Klassen- oder 4-Klassen-Problem und wie stark die Vorverarbeitung dieser Library ist
- each_by_one teilt die Daten in 4 Vektoren auf

https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#module-hrvanalysis.preprocessing
https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#module-hrvanalysis.extract_features

Werte für die oberen und unteren Grenzen habe ich von der Dokumentation übernommen bzw. von diesem Paper
https://openresearchsoftware.metajnl.com/articles/10.5334/jors.305/

Features der Time Domain in den Dictionaries:

- mean_nni: The mean of RR-intervals.
- sdnn : The standard deviation of the time interval between successive normal heart beats (i.e. the RR-intervals).
- sdsd: The standard deviation of differences between adjacent RR-intervals
- rmssd: The square root of the mean of the sum of the squares of differences between adjacent NN-intervals. 
Reflects high frequency (fast or parasympathetic) influences on hrV (i.e., those influencing larger changes from 
one beat to the next).
- median_nni: Median Absolute values of the successive differences between the RR-intervals.
- nni_50: Number of interval differences of successive RR-intervals greater than 50 ms.
- pnni_50: The proportion derived by dividing nni_50 (The number of interval differences of successive RR-intervals 
greater than 50 ms) by the total number of RR-intervals.
- nni_20: Number of interval differences of successive RR-intervals greater than 20 ms.
- pnni_20: The proportion derived by dividing nni_20 (The number of interval differences of successive RR-intervals 
greater than 20 ms) by the total number of RR-intervals.
- range_nni: difference between the maximum and minimum nn_interval.
- cvsd: Coefficient of variation of successive differences equal to the rmssd divided by mean_nni.
- cvnni: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.
- mean_hr: The mean Heart Rate.
- max_hr: Max heart rate.
- min_hr: Min heart rate.
- std_hr: Standard deviation of heart rate.
"""


def feature_extraction_time_domain(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=False):
    detectors = Detectors(fs)

    feature_vector = np.array([])
    targets = np.array([])


    for idx, ecg_lead in enumerate(ecg_leads):
        rr_intervals = detectors.hamilton_detector(ecg_lead)
        if len(rr_intervals) == 1:
            rr_intervals = np.abs(rr_intervals)
            arti_rr_1 = rr_intervals * random.random()
            rr_intervals = np.append(rr_intervals, [arti_rr_1])

        rr_intervals_ms = np.diff(rr_intervals) / fs * 1000  # Umwandlung in ms
        rr_intervals_ms = [abs(number) for number in rr_intervals_ms]

        if nn_intervals:
            rr_intervals_list = get_nn_intervals(rr_intervals=rr_intervals_ms,
                                                 interpolation_method='linear',
                                                 ectopic_beats_removal_method='malik',
                                                 low_rri=300, high_rri=2000)
        else:
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

        if four_problem:
            if ecg_labels[idx] == 'N':
                values_time = list(dict_time_domain.values())
                targets = np.append(targets, 0)
                feature_vector = np.append(feature_vector, values_time)
            if ecg_labels[idx] == 'A':
                values_time = list(dict_time_domain.values())
                targets = np.append(targets, 1)
                feature_vector = np.append(feature_vector, values_time)
            if ecg_labels[idx] == 'O':
                values_time = list(dict_time_domain.values())
                targets = np.append(targets, 2)
                feature_vector = np.append(feature_vector, values_time)
            if ecg_labels[idx] == '~':
                values_time = list(dict_time_domain.values())
                targets = np.append(targets, 3)
                feature_vector = np.append(feature_vector, values_time)
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")

        else:
            if ecg_labels[idx] == 'N':
                values_time = list(dict_time_domain.values())
                targets = np.append(targets, 0)
                feature_vector = np.append(feature_vector, values_time)
            if ecg_labels[idx] == 'A':
                values_time = list(dict_time_domain.values())
                targets = np.append(targets, 1)

                feature_vector = np.append(feature_vector, values_time)
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")

        #idx += 1

    feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 16), 16))

    feature_names = ['mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
                     'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr']
    index = np.arange(len(feature_vector))
    df = pd.DataFrame(data=feature_vector, index=index, columns=feature_names)

    df = df.replace([np.inf, -np.inf], np.nan)
    column_means = df.mean()
    df = df.fillna(column_means)

    df = df.assign(Labels=targets)

    "Normal: 3581, Other: 1713, Noise: 185, AFib: 521"

    feature_vector = df.to_numpy()

    return feature_vector, df


"""
Features der Frequqncy Domain - sollte laut documentation erst bei recordings von 2 - 5 Minuten genutzt werden

Return: Feature Vektoren mit den verschiedenen Größen:

total_power : Total power density spectral
vlf : variance ( = power ) in HRV in the Very low Frequency (.003 to .04 Hz by default). 
Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic activity.
lf : variance ( = power ) in HRV in the low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and 
parasympathetic activity, but in long-term recordings, it reflects sympathetic activity and can be reduced by the 
beta-adrenergic antagonist propanolol.
hf: variance ( = power ) in HRV in the High Frequency (.15 to .40 Hz by default). Reflects fast changes in beat-to-beat 
variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it corresponds to HRV
changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per minute) 
and decreased by anticholinergic drugs or vagal blockade.
lf_hf_ratio : lf/hf ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal balance
lfnu : normalized lf power.
hfnu : normalized hf power.
"""


def feature_extraction_frequency_domain(ecg_leads, ecg_labels, fs, four_problem = False, nn_intervals = False, each_by_one=False):
    detectors = Detectors(fs)

    if each_by_one:
        if four_problem:
            feature_vector_frequency_domain_N = np.array([])
            feature_vector_frequency_domain_A = np.array([])
            feature_vector_frequency_domain_O = np.array([])
            feature_vector_frequency_domain_Noise = np.array([])
        else:
            feature_vector_frequency_domain_N = np.array([])
            feature_vector_frequency_domain_A = np.array([])
    else:
        feature_vector_frequency_domain = np.array([])

    for idx, ecg_lead in enumerate(ecg_leads):
        rr_intervals = detectors.hamilton_detector(ecg_lead)

        if len(rr_intervals) == 1:
            continue

        rr_intervals_ms = np.diff(rr_intervals) / fs * 1000  # Umwandlung in ms

        if nn_intervals:
            rr_intervals_list = get_nn_intervals(rr_intervals=rr_intervals_ms,
                                                 interpolation_method='linear',
                                                 ectopic_beats_removal_method='malik',
                                                 low_rri=300, high_rri=2000)
        else:
            rr_without_outliers = remove_outliers(rr_intervals_ms, low_rri=300, high_rri=2000)
            rr_intervals_list = interpolate_nan_values(rr_without_outliers, interpolation_method='linear')

        rr_intervals_list = [x for x in rr_intervals_list if str(x) != 'nan']  # remove nan values
        if len(rr_intervals_list) <= 2:
            continue

        dict_frequency_domain = hrv.get_frequency_domain_features(rr_intervals_list, sampling_frequency=fs) # muss ggf. angepasst werden für andere Werte

        if each_by_one:
            if four_problem:
                if ecg_labels[idx] == 'N':
                    values = list(dict_frequency_domain.values())
                    feature_vector_frequency_domain_N = np.append(feature_vector_frequency_domain_N, values)
                if ecg_labels[idx] == 'A':
                    values = list(dict_frequency_domain.values())
                    feature_vector_frequency_domain_A = np.append(feature_vector_frequency_domain_A, values)
                if ecg_labels[idx] == 'O':
                    values = list(dict_frequency_domain.values())
                    feature_vector_frequency_domain_O = np.append(feature_vector_frequency_domain_O, values)
                if ecg_labels[idx] == '~':
                    values = list(dict_frequency_domain.values())
                    feature_vector_frequency_domain_Noise = np.append(feature_vector_frequency_domain_Noise, values)
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")

            else:
                if ecg_labels[idx] == 'N':
                    values = list(dict_frequency_domain.values())
                    feature_vector_frequency_domain_N = np.append(feature_vector_frequency_domain_N, values)
                    # print(idx)    #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                elif ecg_labels[idx] == 'A':
                    values = list(dict_frequency_domain.values())
                    feature_vector_frequency_domain_A = np.append(feature_vector_frequency_domain_A, values)
                    # print(idx)     #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")
        else:
            if four_problem:
                if ecg_labels[idx] == 'N':
                    values = list(dict_frequency_domain.values())
                    values.append(0)
                    feature_vector_frequency_domain = np.append(feature_vector_frequency_domain, values)
                if ecg_labels[idx] == 'A':
                    values = list(dict_frequency_domain.values())
                    values.append(1)
                    feature_vector_frequency_domain = np.append(feature_vector_frequency_domain, values)
                if ecg_labels[idx] == 'O':
                    values = list(dict_frequency_domain.values())
                    values.append(2)
                    feature_vector_frequency_domain = np.append(feature_vector_frequency_domain, values)
                if ecg_labels[idx] == '~':
                    values = list(dict_frequency_domain.values())
                    values.append(3)
                    feature_vector_frequency_domain = np.append(feature_vector_frequency_domain, values)
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")

            else:
                if ecg_labels[idx] == 'N':
                    values = list(dict_frequency_domain.values())
                    values.append(0)
                    feature_vector_frequency_domain = np.append(feature_vector_frequency_domain, values)
                    # print(idx)    #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                elif ecg_labels[idx] == 'A':
                    values = list(dict_frequency_domain.values())
                    values.append(1)
                    feature_vector_frequency_domain = np.append(feature_vector_frequency_domain, values)
                    # print(idx)     #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    if each_by_one:
        if four_problem:
            feature_vector_frequency_domain_N = np.reshape(feature_vector_frequency_domain_N,
                                                    (int(len(feature_vector_frequency_domain_N) / 7), 7))
            feature_vector_frequency_domain_A = np.reshape(feature_vector_frequency_domain_A,
                                                    (int(len(feature_vector_frequency_domain_A) / 7), 7))
            feature_vector_frequency_domain_O = np.reshape(feature_vector_frequency_domain_O,
                                                    (int(len(feature_vector_frequency_domain_O) / 7), 7))
            feature_vector_frequency_domain_Noise = np.reshape(feature_vector_frequency_domain_Noise,
                                                      (int(len(feature_vector_frequency_domain_Noise) / 7), 7))
            return feature_vector_frequency_domain_N, feature_vector_frequency_domain_A, feature_vector_frequency_domain_O, \
                   feature_vector_frequency_domain_Noise
        else:
            feature_vector_frequency_domain_N = np.reshape(feature_vector_frequency_domain_N,
                                                    (int(len(feature_vector_frequency_domain_N) / 16), 16))
            feature_vector_frequency_domain_A = np.reshape(feature_vector_frequency_domain_A,
                                                      (int(len(feature_vector_frequency_domain_A) / 16), 16))
            return feature_vector_frequency_domain_N, feature_vector_frequency_domain_A
    else:
         feature_vector_frequency_domain = np.reshape(feature_vector_frequency_domain, (int(len(feature_vector_frequency_domain) / 8), 8))
         return feature_vector_frequency_domain


"""
Geometrical Features - soll erst bei sehr langen Aufnahmen verwendet werden

- triangular_index: The HRV triangular index measurement is the integral of the density distribution 
(= the number of all NN-intervals) divided by the maximum of the density distribution.
- tinn: The triangular interpolation of NN-interval histogram (TINN) is the baseline width of the distribution measured 
as a base of a triangle, approximating the NN-interval distribution
"""


def feature_extraction_geometrical(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=False, each_by_one=False):
    detectors = Detectors(fs)

    if each_by_one:
        if four_problem:
            feature_vector_geometrical_N = np.array([])
            feature_vector_geometrical_A = np.array([])
            feature_vector_geometrical_O = np.array([])
            feature_vector_geometrical_Noise = np.array([])
        else:
            feature_vector_geometrical_N = np.array([])
            feature_vector_geometrical_A = np.array([])
    else:
        feature_vector_geometrical = np.array([])

    for idx, ecg_lead in enumerate(ecg_leads):
        rr_intervals = detectors.hamilton_detector(ecg_lead)
        if len(rr_intervals) == 1:
            continue
        rr_intervals_ms = np.diff(rr_intervals) / fs * 1000  # Umwandlung in ms

        if nn_intervals:
            rr_intervals_list = get_nn_intervals(rr_intervals=rr_intervals_ms,
                                                 interpolation_method='linear',
                                                 ectopic_beats_removal_method='malik',
                                                 low_rri=300, high_rri=2000)
        else:
            rr_without_outliers = remove_outliers(rr_intervals_ms, low_rri=300, high_rri=2000)
            rr_intervals_list = interpolate_nan_values(rr_without_outliers, interpolation_method='linear')

        rr_intervals_list = [x for x in rr_intervals_list if str(x) != 'nan']  # remove nan values
        if len(rr_intervals_list) <= 2:
            continue

        dict_geometrical = hrv.get_geometrical_features(rr_intervals_list)

        if each_by_one:
            if four_problem:
                if ecg_labels[idx] == 'N':
                    values = list(dict_geometrical.values())
                    feature_vector_geometrical_N = np.append(feature_vector_geometrical_N, values)
                if ecg_labels[idx] == 'A':
                    values = list(dict_geometrical.values())
                    feature_vector_geometrical_A = np.append(feature_vector_geometrical_A, values)
                if ecg_labels[idx] == 'O':
                    values = list(dict_geometrical.values())
                    feature_vector_geometrical_O = np.append(feature_vector_geometrical_O, values)
                if ecg_labels[idx] == '~':
                    values = list(dict_geometrical.values())
                    feature_vector_geometrical_Noise = np.append(feature_vector_geometrical_Noise, values)
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")

            else:
                if ecg_labels[idx] == 'N':
                    values = list(dict_geometrical.values())
                    feature_vector_geometrical_N = np.append(feature_vector_geometrical_N, values)
                    # print(idx)    #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                elif ecg_labels[idx] == 'A':
                    values = list(dict_geometrical.values())
                    feature_vector_geometrical_A = np.append(feature_vector_geometrical_A, values)
                    # print(idx)     #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")
        else:
            if four_problem:
                if ecg_labels[idx] == 'N':
                    values = list(dict_geometrical.values())
                    values.append(0)
                    feature_vector_geometrical = np.append(feature_vector_geometrical, values)
                if ecg_labels[idx] == 'A':
                    values = list(dict_geometrical.values())
                    values.append(1)
                    feature_vector_geometrical = np.append(feature_vector_geometrical, values)
                if ecg_labels[idx] == 'O':
                    values = list(dict_geometrical.values())
                    values.append(2)
                    feature_vector_geometrical = np.append(feature_vector_geometrical, values)
                if ecg_labels[idx] == '~':
                    values = list(dict_geometrical.values())
                    values.append(3)
                    feature_vector_geometrical = np.append(feature_vector_geometrical, values)
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")

            else:
                if ecg_labels[idx] == 'N':
                    values = list(dict_geometrical.values())
                    values.append(0)
                    feature_vector_geometrical = np.append(feature_vector_geometrical, values)
                    # print(idx)    #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                elif ecg_labels[idx] == 'A':
                    values = list(dict_geometrical.values())
                    values.append(1)
                    feature_vector_geometrical = np.append(feature_vector_geometrical, values)
                    # print(idx)     #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
                if (idx % 100) == 0:
                    print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    "Reshape of Feature Vectors"
    if each_by_one:
        if four_problem:
            feature_vector_geometrical_N = np.reshape(feature_vector_geometrical_N,
                                                    (int(len(feature_vector_geometrical_N) / 2), 2))
            feature_vector_geometrical_A = np.reshape(feature_vector_geometrical_A,
                                                    (int(len(feature_vector_geometrical_A) / 2), 2))
            feature_vector_geometrical_O = np.reshape(feature_vector_geometrical_O,
                                                    (int(len(feature_vector_geometrical_O) / 2), 2))
            feature_vector_geometrical_Noise = np.reshape(feature_vector_geometrical_Noise,
                                                      (int(len(feature_vector_geometrical_Noise) / 2), 2))
            return feature_vector_geometrical_N, feature_vector_geometrical_A, feature_vector_geometrical_O, \
                   feature_vector_geometrical_Noise
        else:
            feature_vector_geometrical_N = np.reshape(feature_vector_geometrical_N,
                                                    (int(len(feature_vector_geometrical_N) / 2), 2))
            feature_vector_geometrical_A = np.reshape(feature_vector_geometrical_A,
                                                      (int(len(feature_vector_geometrical_A) / 2), 2))
            return feature_vector_geometrical_N, feature_vector_geometrical_A
    else:
         feature_vector_geometrical = np.reshape(feature_vector_geometrical, (int(len(feature_vector_geometrical) / 3), 3))
         return feature_vector_geometrical


"""
Extraction of all kinds of features given by hrv-analysis library

- time domain features
- frequency domain features
- geometrical features
- point-care features
- csi-csv features
- sample entropy 

Target classes values are between [0,3]
0 = Normal
1 = AFib
2 = Other
3 = Noise

"""


def feature_extraction(ecg_leads, ecg_labels, fs, four_problem=False, nn_intervals=False, save=False):
    detectors = Detectors(fs)

    feature_vector = np.array([])
    targets = np.array([])
    cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

    #ecg_leads = Denoise.wavelet(ecg_leads)

    for idx, ecg_lead in enumerate(ecg_leads):
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

        if nn_intervals:
            rr_intervals_list = get_nn_intervals(rr_intervals=rr_intervals_ms,
                                                 interpolation_method='linear',
                                                 ectopic_beats_removal_method='malik',
                                                 low_rri=300, high_rri=2000)
        else:
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

        if four_problem:
            if ecg_labels[idx] == 0: #'N':
                values_time = list(dict_time_domain.values())
                values_frequency = list(dict_frequency_domain.values())
                values_geometrical = list(dict_geometrical_features.values())
                values_pointcare = list(dict_pointcare.values())
                values_entropy = list(dict_entropy.values())
                values_csicsv = list(dict_csi_csv.values())
                targets = np.append(targets, 0)

                feature_vector = np.append(feature_vector, values_time)
                feature_vector = np.append(feature_vector, values_frequency)
                feature_vector = np.append(feature_vector, values_geometrical)
                feature_vector = np.append(feature_vector, values_pointcare)
                feature_vector = np.append(feature_vector, values_entropy)
                feature_vector = np.append(feature_vector, values_csicsv)
                feature_vector = np.append(feature_vector, spectral_features)
            if ecg_labels[idx] == 1: #'A':
                values_time = list(dict_time_domain.values())
                values_frequency = list(dict_frequency_domain.values())
                values_geometrical = list(dict_geometrical_features.values())
                values_pointcare = list(dict_pointcare.values())
                values_entropy = list(dict_entropy.values())
                values_csicsv = list(dict_csi_csv.values())
                targets = np.append(targets, 1)

                feature_vector = np.append(feature_vector, values_time)
                feature_vector = np.append(feature_vector, values_frequency)
                feature_vector = np.append(feature_vector, values_geometrical)
                feature_vector = np.append(feature_vector, values_pointcare)
                feature_vector = np.append(feature_vector, values_entropy)
                feature_vector = np.append(feature_vector, values_csicsv)
                feature_vector = np.append(feature_vector, spectral_features)
            if ecg_labels[idx] == 2: #'O':
                values_time = list(dict_time_domain.values())
                values_frequency = list(dict_frequency_domain.values())
                values_geometrical = list(dict_geometrical_features.values())
                values_pointcare = list(dict_pointcare.values())
                values_entropy = list(dict_entropy.values())
                values_csicsv = list(dict_csi_csv.values())
                targets = np.append(targets, 2)

                feature_vector = np.append(feature_vector, values_time)
                feature_vector = np.append(feature_vector, values_frequency)
                feature_vector = np.append(feature_vector, values_geometrical)
                feature_vector = np.append(feature_vector, values_pointcare)
                feature_vector = np.append(feature_vector, values_entropy)
                feature_vector = np.append(feature_vector, values_csicsv)
                feature_vector = np.append(feature_vector, spectral_features)
            if ecg_labels[idx] == 3: #'~':
                values_time = list(dict_time_domain.values())
                values_frequency = list(dict_frequency_domain.values())
                values_geometrical = list(dict_geometrical_features.values())
                values_pointcare = list(dict_pointcare.values())
                values_entropy = list(dict_entropy.values())
                values_csicsv = list(dict_csi_csv.values())
                targets = np.append(targets, 3)

                feature_vector = np.append(feature_vector, values_time)
                feature_vector = np.append(feature_vector, values_frequency)
                feature_vector = np.append(feature_vector, values_geometrical)
                feature_vector = np.append(feature_vector, values_pointcare)
                feature_vector = np.append(feature_vector, values_entropy)
                feature_vector = np.append(feature_vector, values_csicsv)
                feature_vector = np.append(feature_vector, spectral_features)
            if (idx % 100) == 0:
               print(str(idx) + "\t EKG Signale wurden verarbeitet.")

        else:
            if ecg_labels[idx] == 'N':
                values_time = list(dict_time_domain.values())
                values_frequency = list(dict_frequency_domain.values())
                values_geometrical = list(dict_geometrical_features.values())
                values_pointcare = list(dict_pointcare.values())
                values_entropy = list(dict_entropy.values())
                values_csicsv = list(dict_csi_csv.values())
                targets = np.append(targets, 0)

                feature_vector = np.append(feature_vector, values_time)
                feature_vector = np.append(feature_vector, values_frequency)
                feature_vector = np.append(feature_vector, values_geometrical)
                feature_vector = np.append(feature_vector, values_pointcare)
                feature_vector = np.append(feature_vector, values_entropy)
                feature_vector = np.append(feature_vector, values_csicsv)
            if ecg_labels[idx] == 'A':
                values_time = list(dict_time_domain.values())
                values_frequency = list(dict_frequency_domain.values())
                values_geometrical = list(dict_geometrical_features.values())
                values_pointcare = list(dict_pointcare.values())
                values_entropy = list(dict_entropy.values())
                values_csicsv = list(dict_csi_csv.values())
                targets = np.append(targets, 1)

                feature_vector = np.append(feature_vector, values_time)
                feature_vector = np.append(feature_vector, values_frequency)
                feature_vector = np.append(feature_vector, values_geometrical)
                feature_vector = np.append(feature_vector, values_pointcare)
                feature_vector = np.append(feature_vector, values_entropy)
                feature_vector = np.append(feature_vector, values_csicsv)
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")


    #feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 32), 32))
    feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 57), 57))
    #feature_vector[:,24] = 0

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


    df = df.assign(Labels=targets)

    if save:
        df.to_csv('../datasets/features_uniform_length.csv', encoding='utf-8', index=False)

    "Normal: 3581, Other: 1713, Noise: 185, AFib: 521"

    feature_vector = df.to_numpy()

    return feature_vector, df


#print('Loading dataset')
#df = pd.read_csv('../datasets/df_uniform.csv')
#print('done')
#df = df.to_numpy()
#X = df[:,:-1]
#y = df[:,-1]

#feature_extraction(X, y, fs=300, four_problem=True, nn_intervals=False, save=True)
"""
Random Oversampling method
"""
def synthesize_data_naive(df):
    normal, afib, other, noise = df['Labels'].value_counts()
    AFib = df[df['Labels'] == 1]
    Noise = df[df['Labels'] == 3]

    AFib_over = AFib.sample(other, replace=True)
    Noise_over = Noise.sample(other, replace=True)

    df_synth = pd.concat([df, AFib_over], axis=0)
    df_synth = pd.concat([df_synth, Noise_over], axis=0)

    features_synth = df_synth.to_numpy()
    return features_synth


"SMOTE Algorithm to handle class imbalanace"
def smote_algo(X, y):
    y = LabelEncoder().fit_transform(y)
    sm = SMOTE(random_state=42)
    X_synth, y_synth = sm.fit_resample(X, y)
    return X_synth, y_synth


"Creates a pandas dataframe"
def create_dataset(X, y, save=False):
    #feature_names = ['mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
    #                'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'total_power', 'vlf', 'lf',
    #                 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'triangular_index', 'tinn', 'sd1', 'sd2', 'ratio_sd2_sd1',
    #                 'csi', 'cvi', 'Modified_csi', 'sampen']
    feature_names = ['mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
                    'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'total_power', 'vlf', 'lf',
                     'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'triangular_index', 'sd1', 'sd2', 'ratio_sd2_sd1',
                     'csi', 'cvi', 'Modified_csi', 'sampen', 'Fundamental freq', 'Human energy range',
                     'Max power spectrum', 'Max Frequency', 'Median Frequency', 'Power bandwith',
                     'Spectral centroid', 'Spectral decrease', 'Spectral entropy', 'Spectral kurtosis',
                     'Spectral positive turning points', 'Spectral roll-off', 'Spectral roll-on', 'Spectral skewness',
                     'Spectral spread', 'Spectral variation', 'Wavelet abs mean 1', 'Wavelet abs mean 2',
                     'Wavelet abs mean 3', 'Wavelet abs mean 4', 'Wavelet abs mean 5', 'Wavelet abs mean 6',
                     'Wavelet abs mean 7', 'Wavelet abs mean 8', 'Wavelet abs mean 9']
    index = np.arange(len(X))
    df = pd.DataFrame(data=X, index=index, columns=feature_names)
    df = df.assign(Labels=y)

    if save:
        df.to_csv('../datasets/two_average_filtered_synth_extended.csv', encoding='utf-8', index=False)
    return df


def histoplot(heart_rate, bins):
    plt.hist(heart_rate, bins, range=[min(heart_rate),max(heart_rate)])
    plt.xlabel('Heart rate')
    plt.ylabel('Number of Samples')
    plt.show()
    mean_hr = np.nanmean(heart_rate)
    max_hr = max(heart_rate)
    min_hr = min(heart_rate)
    print("HR-min: ", min_hr, "HR-max: ", max_hr, "HR-mean: ", mean_hr)


def hr_analysis(dictionary_list, bins=50, is_fourproblem=False):
    if is_fourproblem:
        for i in range(3):
            heart_rate = []
            for j in range(len(dictionary_list)):
                dict_temp = dictionary_list[i]
                heart_rate.append(dict_temp['mean_hr'])
            histoplot(heart_rate, bins)
    else:
        for i in range(1):
            heart_rate = []
            for j in range(len(dictionary_list)):
                dict_temp = dictionary_list[i]
                heart_rate.append(dict_temp['mean_hr'])
            histoplot(heart_rate, bins)

"Extract spectral features of ecg signal"
def tsfel_features(ecg_leads):
    cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

    features = np.array([])
    for idx, ecg_lead in enumerate(ecg_leads):
    #for lead in range(len(ecg_leads)):
        #ecg_lead = ecg_leads[lead]
        X = tsfel.time_series_features_extractor(cfg, ecg_lead, fs=300)
        corr_features = tsfel.correlated_features(X)
        X.drop(corr_features, axis=1, inplace=True)
        X = X.to_numpy()
        features = np.append(features, X)
        #features.append(X)
        if (idx % 100) == 0:
            print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    features = np.reshape(features, (int(len(features) / 25), 25))
    return features


def uniform_length(ecg_leads, ecg_labels):
    ecg_labels_std = []
    ecg_leads_std = []
    ecg_leads_extra = []
    ecg_labels_extra = []

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
            lowiter= 9000//len(ecg_leads[index])
            print(lowiter)
            for i in range(lowiter):
                print(ecg_leads[index].shape)
                ecg_leads[index]=np.append(ecg_leads[index],ecg_leads[index])
                print('dadadad',ecg_leads[index].shape)
            ecg_leads[index]=ecg_leads[index][0:9000]
            print(len(ecg_leads[index]))
        elif len(ecg_leads[index]>9000):
            if len(ecg_leads[index]<=18000):
                ecg_leads[index]=ecg_leads[index][0:9000]
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                ecg_labels_extra.append(ecg_labels_std[index])
            elif len(ecg_leads[index]>18000):
                iter = len(ecg_leads[index])//9000
                ecg_leads[index]=ecg_leads[index][:9000]
                for i in range(1,iter):
                    start = 9000*i
                    end = 9000*(i+1)
                    ecg_leads_extra.append(ecg_leads[start:end])
                    ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                ecg_labels_extra.append(ecg_labels_std[index])

    ecg_labels_std = ecg_labels_std + ecg_labels_extra
    ecg_leads_std = ecg_leads + ecg_leads_extra

    col = np.arange(9000)

    lst =[]
    for idx, ecg_lead in enumerate(ecg_leads_std):
        lst.append(ecg_lead)
        if (idx % 100) == 0:
            print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    df = pd.DataFrame(lst, columns=col)
    df = df.assign(Labels=ecg_labels_std)

    df = df.to_numpy()
    return df

def uniform_dataframe(ecg_leads, ecg_labels, save=False):
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



#df = uniform_length(ecg_leads, ecg_labels)

#X = df[:,:-1]
#y = df[:,-1]
#X_synth, y_synth = smote_algo(X, y)
#print('Resampled Dataset shape %s' % Counter(y_synth))


#features = tsfel_features(ecg_leads)
#print(features[0].head())
#spectral_features = ['Fundamental freq', 'Human energy range', 'Max power spectrum', 'Max Frequency',
#                     'Median Frequency', 'Power bandwith', 'Spectral centroid', 'Spectral decrease',
#                     'Spectral entropy', 'Spectral kurtosis', 'Spectral positive turning points',
#                     'Spectral roll-off', 'Spectral roll-on', 'Spectral skewness',  'Spectral spread',
#                     'Spectral variation', 'Wavelet abs mean 1', 'Wavelet abs mean 2','Wavelet abs mean 3',
#                     'Wavelet abs mean 4', 'Wavelet abs mean 5', 'Wavelet abs mean 6', 'Wavelet abs mean 7',
#                     'Wavelet abs mean 8' , 'Wavelet abs mean 9']
#idx = np.arange(len(features))
#df = pd.DataFrame(data=features, index=idx, columns=spectral_features)
#df.head()

#features, df = feature_extraction(ecg_leads, ecg_labels, fs, four_problem=True, save=True)
#print(df.head())
#df = pd.read_csv('features_filtered_extended.csv')
#df = df.to_numpy()


#X = features[:,:-1]
#y = features[:,-1]
#X_synth, y_synth = smote_algo(X, y)
#print('Resampled Dataset shape %s' % Counter(y_synth))

#df = create_dataset(X_synth, y_synth, save=False)
#print('done')

