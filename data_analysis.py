"""
Datei ist für Analyse der Daten gedacht. Welche Sachen sind auffällig, wie können die Klassen unterschieden werden.
"""

import numpy as np
import matplotlib.pyplot as plt
from wettbewerb import load_references
from collections import Counter
from scipy.signal import medfilt
from ecgdetectors import Detectors
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_geometrical_features
from hrvanalysis.preprocessing import get_nn_intervals, remove_outliers, interpolate_nan_values

ecg_leads, ecg_labels, fs, ecg_names = load_references()
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
    kernel_1 = 0.2/(1/fs)    #Wurden auf 200 und 600ms gesetzt nach dem Paper
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
- Frequency domain aber eigentlich nur bei Samples zwischen 2-5 Minuten, deshalb auskommentiert
- Features werden pro Iteration in einem Dictionary zurückgegeben und in einer Liste gespeichert
- Flags geben an, ob 2-Klassen- oder 4-Klassen-Problem und wie stark die Vorverarbeitung dieser Library ist

https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#module-hrvanalysis.preprocessing
https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#module-hrvanalysis.extract_features

Werte für die oberen und unteren Grenzen habe ich von der Dokumentation übernommen bzw. von diesem Paper
https://openresearchsoftware.metajnl.com/articles/10.5334/jors.305/

Features der Time Domain in den Dictionaries:

- mean_nni: The mean of RR-intervals.
- sdnn : The standard deviation of the time interval between successive normal heart beats (i.e. the RR-intervals).
- sdsd: The standard deviation of differences between adjacent RR-intervals
- rmssd: The square root of the mean of the sum of the squares of differences between adjacent NN-intervals. 
Reflects high frequency (fast or parasympathetic) influences on hrV (i.e., those influencing larger changes from one beat to the next).
- median_nni: Median Absolute values of the successive differences between the RR-intervals.
- nni_50: Number of interval differences of successive RR-intervals greater than 50 ms.
- pnni_50: The proportion derived by dividing nni_50 (The number of interval differences of successive RR-intervals greater than 50 ms) by the total number of RR-intervals.
- nni_20: Number of interval differences of successive RR-intervals greater than 20 ms.
- pnni_20: The proportion derived by dividing nni_20 (The number of interval differences of successive RR-intervals greater than 20 ms) by the total number of RR-intervals.
- range_nni: difference between the maximum and minimum nn_interval.
- cvsd: Coefficient of variation of successive differences equal to the rmssd divided by mean_nni.
- cvnni: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.
- mean_hr: The mean Heart Rate.
- max_hr: Max heart rate.
- min_hr: Min heart rate.
- std_hr: Standard deviation of heart rate.
"""
def feature_extraction_td(ecg_leads, ecg_labels, fs, four_problem = False, nn_intervals = False):
    detectors = Detectors(fs)

    if four_problem == True:
        dict_list_time_domain_N = []
        dict_list_time_domain_A = []
        dict_list_time_domain_O = []
        dict_list_time_domain_Noise = []
    else:
        dict_list_time_domain_N = []
        dict_list_time_domain_Other = []


    for idx, ecg_lead in enumerate(ecg_leads):
        rr_intervals = detectors.hamilton_detector(ecg_lead)
        if len(rr_intervals) == 1:
            continue
        rr_intervals_ms = np.diff(rr_intervals)/fs*1000 # Umwandlung in ms

        if nn_intervals:
            rr_intervals_list = get_nn_intervals(rr_intervals=rr_intervals_ms,
                                                interpolation_method='linear',
                                                ectopic_beats_removal_method='malik',
                                                low_rri=300, high_rri=2000)
        else:
            rr_without_outliers = remove_outliers(rr_intervals_ms, low_rri=300, high_rri=2000)
            rr_intervals_list = interpolate_nan_values(rr_without_outliers, interpolation_method='linear')
        if len(rr_intervals_list) <= 2:
            continue
        dict_time_domain = get_time_domain_features(rr_intervals_list)

        if four_problem == True:
            if ecg_labels[idx] == 'N':
                dict_copy = dict_time_domain.copy()
                dict_list_time_domain_N.append(dict_copy)
            if ecg_labels[idx] == 'A':
                dict_copy = dict_time_domain.copy()
                dict_list_time_domain_A.append(dict_copy)
            if ecg_labels[idx] == 'O':
                dict_copy = dict_time_domain.copy()
                dict_list_time_domain_O.append(dict_copy)
            if ecg_labels[idx] == '~':
                dict_copy = dict_time_domain.copy()
                dict_list_time_domain_Noise.append(dict_copy)
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")

        else:
            if ecg_labels[idx] == 'N':
                dict_copy = dict_time_domain.copy()
                dict_list_time_domain_N.append(dict_copy)
                #print(idx)    #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
            else:
                dict_copy = dict_time_domain.copy()
                dict_list_time_domain_Other.append(dict_copy)
                #print(idx)     #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")


    if four_problem:
        dictionary_list = [dict_list_time_domain_N, dict_list_time_domain_A, dict_list_time_domain_O,
                           dict_list_time_domain_Noise]
        return dictionary_list
    else:
        dictionary_list = [dict_list_time_domain_N, dict_list_time_domain_Other]
        return dictionary_list


"""
Features der Frequqncy Domain - sollte laut documentation erst bei recordings von 2 - 5 Minuten genutzt werden

Return: Dictionaries mit den verschiedenen Größen:

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
def feature_extraction_fd(ecg_leads, ecg_labels, fs, four_problem = False, nn_intervals = False):   #
    detectors = Detectors(fs)

    if four_problem == True:
        dict_list_frequency_domain_N = []
        dict_list_frequency_domain_A = []
        dict_list_frequency_domain_O = []
        dict_list_frequency_domain_Noise = []
    else:
        dict_list_frequency_domain_N = []
        dict_list_frequency_domain_Other = []

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
        if len(rr_intervals_list) <= 2:
            continue

        dict_frequency_domain = get_frequency_domain_features(rr_intervals_list, sampling_frequency=fs) # muss ggf. angepasst werden für andere Werte

        if four_problem == True:
            if ecg_labels[idx] == 'N':
                dict_copy = dict_frequency_domain.copy()
                dict_list_frequency_domain_N.append(dict_copy)
            if ecg_labels[idx] == 'A':
                dict_copy = dict_frequency_domain.copy()
                dict_list_frequency_domain_A.append(dict_copy)
            if ecg_labels[idx] == 'O':
                dict_copy = dict_frequency_domain.copy()
                dict_list_frequency_domain_O.append(dict_copy)
            if ecg_labels[idx] == '~':
                dict_copy = dict_frequency_domain.copy()
                dict_list_frequency_domain_Noise.append(dict_copy)
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")

        else:
            if ecg_labels[idx] == 'N':
                dict_copy = dict_frequency_domain.copy()
                dict_list_frequency_domain_N.append(dict_copy)
                #print(idx)    #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
            else:
                dict_copy = dict_frequency_domain.copy()
                dict_list_frequency_domain_Other.append(dict_copy)
                #print(idx)     #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    if four_problem:
        dictionary_list = [dict_list_frequency_domain_N, dict_list_frequency_domain_A, dict_list_frequency_domain_O,
                           dict_list_frequency_domain_Noise]
        return dictionary_list
    else:
        dictionary_list = [dict_list_frequency_domain_N, dict_list_frequency_domain_Other]
        return dictionary_list

def feature_extraction_geometrical(ecg_leads, ecg_labels, fs, four_problem = False, nn_intervals = False):
    detectors = Detectors(fs)

    if four_problem == True:
        dict_list_geometrical_N = []
        dict_list_geometrical_A = []
        dict_list_geometrical_O = []
        dict_list_geometrical_Noise = []
    else:
        dict_list_geometrical_N = []
        dict_list_geometrical_Other = []

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
        if len(rr_intervals_list) <= 2:
            continue
        dict_time_domain = get_geometrical_features(rr_intervals_list)

        if four_problem == True:
            if ecg_labels[idx] == 'N':
                dict_copy = dict_time_domain.copy()
                dict_list_geometrical_N.append(dict_copy)
            if ecg_labels[idx] == 'A':
                dict_copy = dict_time_domain.copy()
                dict_list_geometrical_A.append(dict_copy)
            if ecg_labels[idx] == 'O':
                dict_copy = dict_time_domain.copy()
                dict_list_geometrical_O.append(dict_copy)
            if ecg_labels[idx] == '~':
                dict_copy = dict_time_domain.copy()
                dict_list_geometrical_Noise.append(dict_copy)
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")

        else:
            if ecg_labels[idx] == 'N':
                dict_copy = dict_time_domain.copy()
                dict_list_geometrical_N.append(dict_copy)
                # print(idx)    #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
            else:
                dict_copy = dict_time_domain.copy()
                dict_list_geometrical_Other.append(dict_copy)
                # print(idx)     #zum debuggen, mit welchem Sample die Methode nicht zurechtkommt
            if (idx % 100) == 0:
                print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    if four_problem:
        dictionary_list = [dict_list_geometrical_N, dict_list_geometrical_A, dict_list_geometrical_O,
                           dict_list_geometrical_Noise]
        return dictionary_list
    else:
        dictionary_list = [dict_list_geometrical_N, dict_list_geometrical_Other]
        return dictionary_list

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
