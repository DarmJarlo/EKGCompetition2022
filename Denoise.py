
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.signal import medfilt
from scipy import signal
import os
##https://blog.csdn.net/qq_36495569/article/details/104086636
#https://blog.csdn.net/qq_39594939/article/details/115697198
def wavelet(data):

    data_denoise = []
    for i in range(len(data)):

        w = pywt.Wavelet('db8')  # Daubechies8
        maxlev = pywt.dwt_max_level(len(data[i]), w.dec_len)
        print("maximum level is " + str(maxlev))
        threshold = 0.08  # Threshold for filtering
        coeffs = pywt.wavedec(data[i], 'db8', level = maxlev)  # decomposition


        for j in range(1, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))  # 将噪声filter

        datarec = pywt.waverec(coeffs, 'db8')  # reconstruct with small waves
        data_denoise.append(datarec)

    return data_denoise

def normalize(data):
    data = data.astype('float')
    mx = np.max(data).astype(np.float64)
    mn = np.min(data).astype(np.float64)
    return np.true_divide(data-mn,mx-mn,out =np.zeros_like(data-mn),where=(mx-mn)!=0)


def median_filter(data):
    data_denoise = []

    for i in range(len(data)):
        data1 = normalize(data[i])
        data2 = medfilt(data1,3)
        data3 = medfilt(data2,5)
        data_de= data1-data3
        data_denoise.append(data3)
    return data_denoise


def butterworth(data):
    data_denoise = []
    for i in range(len(data)):
        b, a = signal.butter(8,0.1, 'lowpass') #lowpass could also be bandpass or highpass
        data1 = signal.filtfilt(b,a,data[i])
        data_denoise.append(data1)
    return data_denoise




def compare_plot(data,datarec):
    mintime = 0
    maxtime = mintime + len(data) + 1
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data[300:800])
    #plt.xlabel(‘time(s)’)

    plt.title("Raw signal")
    plt.subplot(2, 1, 2)
    plt.plot(datarec[300:800])
   # plt.xlabel(‘time(s)’)

    plt.title("wavelet")

    plt.tight_layout()
    plt.show()

