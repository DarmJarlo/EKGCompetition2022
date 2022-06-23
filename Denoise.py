
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.signal import medfilt
from scipy import signal
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import os
##https://blog.csdn.net/qq_36495569/article/details/104086636
#https://blog.csdn.net/qq_39594939/article/details/115697198

def relength(ecg_leads,ecg_labels):
    ecg_labels_std = []
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
            lowiter = 9000 // len(ecg_leads[index])
            print(lowiter)
            for i in range(lowiter):
                print(ecg_leads[index].shape)
                ecg_leads[index] = np.append(ecg_leads[index], ecg_leads[index])
                print('dadadad', ecg_leads[index].shape)
            ecg_leads[index] = ecg_leads[index][0:9000]
            print(len(ecg_leads[index]))
        elif len(ecg_leads[index] > 9000):
            if len(ecg_leads[index] <= 18000):
                ecg_leads[index] = ecg_leads[index][0:9000]
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                ecg_labels_extra.append(ecg_labels_std[index])
            elif len(ecg_leads[index] > 18000):
                iter = len(ecg_leads[index]) // 9000
                ecg_leads[index] = ecg_leads[index][:9000]
                for i in range(1, iter):
                    start = 9000 * i
                    end = 9000 * (i + 1)
                    ecg_leads_extra.append(ecg_leads[start:end])
                    ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                ecg_labels_extra.append(ecg_labels_std[index])

    ecg_labels_std = ecg_labels_std + ecg_labels_extra
    ecg_leads_std = ecg_leads + ecg_leads_extra

    print(ecg_labels_extra)
    # form the label as one-hot

    Label_set = np.zeros((len(ecg_labels_std), 4))

    for i in range(len(ecg_labels_std)):
        print('111111', i, ecg_labels_std[i])
        dummy = np.zeros(4)
        dummy[int(ecg_labels_std[i])] = 1
        Label_set[i, :] = dummy
    return ecg_leads_std,Label_set[i,:]


def wavelet(data,label):

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

"SMOTE Algorithm to handle class imbalanace"
def smote_algo(X, y):
    y = LabelEncoder().fit_transform(y)
    sm = SMOTE(random_state=42)
    X_synth, y_synth = sm.fit_resample(X, y)
    return X_synth, y_synth


