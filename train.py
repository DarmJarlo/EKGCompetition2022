# -*- coding: utf-8 -*-
"""
Model 2: XGBoost Classifier

Using the hrvanalysis library given by: https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#
+ tsfel library given by: https://github.com/fraunhoferportugal/tsfel and
https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html

Features were extracted via the different methods:
(descriptions were copied from:
https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#module-hrvanalysis.extract_features)

- time domain features
    mean_nni: The mean of RR-intervals.
    sdnn : The standard deviation of the time interval between successive normal heart beats (i.e. the RR-intervals).
    sdsd: The standard deviation of differences between adjacent RR-intervals
    rmssd: The square root of the mean of the sum of the squares of differences between adjacent NN-intervals.
    Reflects high frequency (fast or parasympathetic) influences on hrV (i.e., those influencing larger changes from
    one beat to the next).
    median_nni: Median Absolute values of the successive differences between the RR-intervals.
    nni_50: Number of interval differences of successive RR-intervals greater than 50 ms.
    pnni_50: The proportion derived by dividing nni_50 (The number of interval differences of successive RR-intervals
    greater than 50 ms) by the total number of RR-intervals.
    nni_20: Number of interval differences of successive RR-intervals greater than 20 ms.
    pnni_20: The proportion derived by dividing nni_20 (The number of interval differences of successive RR-intervals
    greater than 20 ms) by the total number of RR-intervals.
    range_nni: difference between the maximum and minimum nn_interval.
    cvsd: Coefficient of variation of successive differences equal to the rmssd divided by mean_nni.
    cvnni: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.
    mean_hr: The mean Heart Rate.
    max_hr: Max heart rate.
    min_hr: Min heart rate.
    std_hr: Standard deviation of heart rate
- frequency domain features
    total_power: Total power density spectral
    vlf: variance ( = power ) in HRV in the Very low Frequency (.003 to .04 Hz by default).
    Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic activity.
    lf: variance ( = power ) in HRV in the low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and
    parasympathetic activity, but in long-term recordings, it reflects sympathetic activity and can be reduced by the
    beta-adrenergic antagonist propanolol.
    hf: variance ( = power ) in HRV in the High Frequency (.15 to .40 Hz by default). Reflects fast changes in
    beat-to-beat variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it
    corresponds to HRV changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or
    7 breaths per minute) and decreased by anticholinergic drugs or vagal blockade.
    lf_hf_ratio: lf/hf ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal
    balance.
    lfnu: normalized lf power.
    hfnu: normalized hf power.
- geometrical features
    triangular_index: The HRV triangular index measurement is the integral of the density distribution
    (= the number of all NN-intervals) divided by the maximum of the density distribution.
    tinn: The triangular interpolation of NN-interval histogram (TINN) is the baseline width of the distribution
    measured as a base of a triangle, approximating the NN-interval distribution
- pointcare plot features
    sd1 : The standard deviation of projection of the Poincaré plot on the line perpendicular to the line of identity.
    sd2 : SD2 is defined as the standard deviation of the projection of the Poincaré plot on the line of identity (y=x)
    ratio_sd2_sd1 : Ratio between SD2 and SD1.
- csi cvi features
    csi : Cardiac Sympathetic Index.
    cvi : Cadiac Vagal Index.
    Modified_csi : Modified CSI is an alternative measure in research of seizure detection.
- sample entropy of the data

- spectral domain with provided features.json (needed for tsfel library)


model will be saved as 'model_4p_2.npy'
"""

import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references
import hrvanalysis as hrv
from hrvanalysis.preprocessing import remove_outliers, interpolate_nan_values
import random
import pandas as pd
import tsfel
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads, ecg_labels, fs, ecg_names = load_references()

detectors = Detectors(fs)
cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

feature_vector = np.array([])  # create empty arrays for features and targets
targets = np.array([])

for idx, ecg_lead in enumerate(ecg_leads):
    spectral_features = tsfel.time_series_features_extractor(cfg, ecg_lead, fs=fs)
    corr_features = tsfel.correlated_features(spectral_features)
    spectral_features.drop(corr_features, axis=1, inplace=True)
    spectral_features = spectral_features.to_numpy()

    rr_intervals = detectors.two_average_detector(ecg_lead)

    if len(rr_intervals) == 1:
        rr_intervals = np.abs(rr_intervals)
        arti_rr_1 = rr_intervals * random.random()
        rr_intervals = np.append(rr_intervals, [arti_rr_1])     # append rr_intervals to have at least two values

    rr_intervals_ms = np.diff(rr_intervals) / fs * 1000  # Umwandlung in ms
    rr_intervals_ms = [abs(number) for number in rr_intervals_ms]  # make rr_intervals positive

    rr_without_outliers = remove_outliers(rr_intervals_ms, low_rri=300, high_rri=2000)  # preprocessing
    rr_intervals_list = interpolate_nan_values(rr_without_outliers, interpolation_method='linear')

    rr_intervals_list = [x for x in rr_intervals_list if str(x) != 'nan']  # remove nan values

    if len(rr_intervals_list) <= 2:       # frequency domain needs at least 3 values
        mean_rr = np.nanmean(rr_intervals_list)
        rr_intervals_list = np.nan_to_num(rr_intervals, nan=mean_rr)
        arti_rr_1 = rr_intervals_list[0] * random.random()
        arti_rr_2 = rr_intervals_list[0] * random.random()
        rr_intervals_list = np.append(rr_intervals_list, arti_rr_1)
        rr_intervals_list = np.append(rr_intervals_list, arti_rr_2)

    dict_time_domain = hrv.get_time_domain_features(rr_intervals_list)  # feature extraction via hrv
    dict_geometrical_features = hrv.get_geometrical_features(rr_intervals_list)
    dict_pointcare = hrv.get_poincare_plot_features(rr_intervals_list)
    dict_csi_csv = hrv.get_csi_cvi_features(rr_intervals_list)
    dict_entropy = hrv.get_sampen(rr_intervals_list)
    dict_frequency_domain = hrv.get_frequency_domain_features(rr_intervals_list)

    if ecg_labels[idx] == 'N':
        values_time = list(dict_time_domain.values())
        values_frequency = list(dict_frequency_domain.values())
        values_geometrical = list(dict_geometrical_features.values())
        values_pointcare = list(dict_pointcare.values())
        values_entropy = list(dict_entropy.values())
        values_csicsv = list(dict_csi_csv.values())
        targets = np.append(targets, 0)             # saves 'N' as 0

        feature_vector = np.append(feature_vector, values_time)
        feature_vector = np.append(feature_vector, values_frequency)
        feature_vector = np.append(feature_vector, values_geometrical)
        feature_vector = np.append(feature_vector, values_pointcare)
        feature_vector = np.append(feature_vector, values_entropy)
        feature_vector = np.append(feature_vector, values_csicsv)
        feature_vector = np.append(feature_vector, spectral_features)
    if ecg_labels[idx] == 'A':
        values_time = list(dict_time_domain.values())
        values_frequency = list(dict_frequency_domain.values())
        values_geometrical = list(dict_geometrical_features.values())
        values_pointcare = list(dict_pointcare.values())
        values_entropy = list(dict_entropy.values())
        values_csicsv = list(dict_csi_csv.values())
        targets = np.append(targets, 1)             # saves 'A' as 1

        feature_vector = np.append(feature_vector, values_time)
        feature_vector = np.append(feature_vector, values_frequency)
        feature_vector = np.append(feature_vector, values_geometrical)
        feature_vector = np.append(feature_vector, values_pointcare)
        feature_vector = np.append(feature_vector, values_entropy)
        feature_vector = np.append(feature_vector, values_csicsv)
        feature_vector = np.append(feature_vector, spectral_features)
    if ecg_labels[idx] == 'O':
        values_time = list(dict_time_domain.values())
        values_frequency = list(dict_frequency_domain.values())
        values_geometrical = list(dict_geometrical_features.values())
        values_pointcare = list(dict_pointcare.values())
        values_entropy = list(dict_entropy.values())
        values_csicsv = list(dict_csi_csv.values())
        targets = np.append(targets, 2)             # saves 'O' as 2

        feature_vector = np.append(feature_vector, values_time)
        feature_vector = np.append(feature_vector, values_frequency)
        feature_vector = np.append(feature_vector, values_geometrical)
        feature_vector = np.append(feature_vector, values_pointcare)
        feature_vector = np.append(feature_vector, values_entropy)
        feature_vector = np.append(feature_vector, values_csicsv)
        feature_vector = np.append(feature_vector, spectral_features)
    if ecg_labels[idx] == '~':
        values_time = list(dict_time_domain.values())
        values_frequency = list(dict_frequency_domain.values())
        values_geometrical = list(dict_geometrical_features.values())
        values_pointcare = list(dict_pointcare.values())
        values_entropy = list(dict_entropy.values())
        values_csicsv = list(dict_csi_csv.values())
        targets = np.append(targets, 3)             # saves '~' as 3

        feature_vector = np.append(feature_vector, values_time)
        feature_vector = np.append(feature_vector, values_frequency)
        feature_vector = np.append(feature_vector, values_geometrical)
        feature_vector = np.append(feature_vector, values_pointcare)
        feature_vector = np.append(feature_vector, values_entropy)
        feature_vector = np.append(feature_vector, values_csicsv)
        feature_vector = np.append(feature_vector, spectral_features)
    if (idx % 100) == 0:
        print(str(idx) + "\t EKG Signale wurden verarbeitet.")


feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 57), 57))  # reshape fv


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
df = df.drop(columns=['tinn'])  # column has None-values

df = df.replace([np.inf, -np.inf], np.nan)   # Replace other invalid values
column_means = df.mean()
df = df.fillna(column_means)

df = df.assign(Labels=targets)

df = df.to_numpy()
X = df[:, :-1]
y = df[:, -1]

y = LabelEncoder().fit_transform(y)
sm = SMOTE(random_state=42)     # handle class_imbalance using oversampling via SMOTE
X, y = sm.fit_resample(X, y)

xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6, min_child_weight=0, gamma=0,
                          subsample=0.55, colsample_bytree=0.75, bjective='multi:softmax',
                          nthread=4, scale_pos_weight=1, seed=42)
xgb.fit(X, y)                # fit XGBoost Classifier



if os.path.exists("model_4p_2.npy"):
    os.remove("model_4p_2.npy")
with open('model_4p_2.npy', 'wb') as f:
    pickle.dump(xgb, f)          # save model

print('Training is done')
