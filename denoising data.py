"""
File for denoising and smoothing the ecg data
"""

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from wettbewerb import load_references

ecg_leads, ecg_labels, fs, ecg_names = load_references()
times = np.arange(len(ecg_leads[0]))/fs

b, a = scipy.signal.butter(3, 0.1)
filtered = []
for idx, ecg_lead in enumerate(ecg_leads):
    filtered.append(scipy.signal.filtfilt(b, a, ecg_lead))
print(filtered)

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(times, ecg_leads[0])
plt.title("ECG Signal with Noise")
plt.margins(0, .05)

plt.subplot(122)
plt.plot(times, filtered[0])
plt.title("Filtered ECG Signal")
plt.margins(0, .05)

plt.tight_layout()
plt.show()