# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images

class Locate():
    def __init__(self):
        self.R_peak = 0
        # Retrieve ECG data from data folder (sampling rate= 1000 Hz)

        # Extract R-peaks locations
    def Locate_R(self,ecg_data):
        _, rpeaks = nk.ecg_peaks(ecg_data, sampling_rate=3000)
        print(rpeaks)
        # if we cannot find the peak by this approach ,we may do the gradient search
        # in 1 2 3 4 order or in 5 4 3 2 1 order to find the first peak and the last peak
        return rpeaks

