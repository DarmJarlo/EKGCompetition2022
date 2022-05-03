"""
Datei ist für Analyse der Daten gedacht. Welche Sachen sind auffällig, wie können die Klassen unterschieden werden.
"""

import numpy as np
import matplotlib.pyplot as plt
from wettbewerb import load_references
from collections import Counter

ecg_leads, ecg_labels, fs, ecg_names = load_references()
#a = dict(Counter(ecg_labels))
#print(a)
"Normal: 3581, Other: 1713, Noise: 185, AFib: 521"

"Cropping the data"
size = []
for i in range(len(ecg_leads)):
    ecg_lead = ecg_leads[i]
    size.append(len(ecg_lead))

smallest_length = min(size)  # smallest sample = 2714

"Standardize the data"
means = []
stds = []
for i in range(len(ecg_leads)):
    ecg_leads[i] = ecg_leads[i][:smallest_length]
    mean = np.mean(ecg_leads[i])
    std = np.std(ecg_leads[i])
    means.append(mean)
    stds.append(std)

data_mean = np.mean(means)
data_std = np.mean(stds)

ecg_leads_standardized = [] #list of the standardized samples
for i in range(len(ecg_leads)):
    new_ecgs = (ecg_leads[i] - data_mean) / data_std
    ecg_leads_standardized.append(new_ecgs)