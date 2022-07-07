"""
Helper Functions for train/predict
"""


from wettbewerb import load_references
from collections import Counter
from scipy.signal import medfilt
from ecgdetectors import Detectors
#import hrvanalysis as hrv
#from hrvanalysis.preprocessing import get_nn_intervals, remove_outliers, interpolate_nan_values
import random
#import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import pywt
import numpy as np
from scipy.signal import medfilt
from scipy import signal
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.signal import medfilt
from scipy import signal
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from biosppy.signals.ecg import christov_segmenter
import neurokit2 as nk
import os
##https://blog.csdn.net/qq_36495569/article/details/104086636
#https://blog.csdn.net/qq_39594939/article/details/115697198
"""
SMOTE Algorithm to handle class imbalanace
"""


def smote_algo(X, y):
    y = LabelEncoder().fit_transform(y)
    sm = SMOTE(random_state=42)
    X_synth, y_synth = sm.fit_resample(X, y)
    return X_synth, y_synth

"""
make samples to length 9000
"""
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
                ecg_tem=ecg_leads[index][:9000]
                for i in range(1,iter):
                    start = 9000*i
                    end = 9000*(i+1)
                    ecg_leads_extra.append(ecg_leads[start:end])
                    ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads[index] = ecg_tem

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


def Locate_R(ecg,sampling_rate=300):
    '''locate r peak '''
    rpeaks = christov_segmenter(signal=ecg,sampling_rate=sampling_rate)
    #print(rpeaks)
    return rpeaks[0]
def relength(ecg_leads,ecg_labels):
    '''make the labels as one hot and make the leads the same length with 9000
    short data : duplicate the part between the first R peak and the last R peak until 9000, if no peak detected,
    directly duplicate
    long data :sliding window with size 9000'''
    ecg_labels_std = []
    ecg_leads_extra = []
    ecg_labels_extra = []
    extra_index = []

    n=len(ecg_leads)
    index_plus = n - 1
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
            Rpeak = Locate_R(ecg_leads[index])
            if len(Rpeak)>1:

                ecg_leads[index]=ecg_leads[index][Rpeak[0]:Rpeak[-1]]
                #print("rpeaks",Rpeak[0],Rpeak[-1],len(ecg_leads[index]))
                lowiter = 9000 // len(ecg_leads[index])

            else:
                lowiter = 9000 // len(ecg_leads[index])
                #print(lowiter)


            ecg_temp=ecg_leads[index]
            for i in range(lowiter):
                #print(ecg_leads[index].shape)
                ecg_temp = np.append(ecg_temp, ecg_leads[index])
            ecg_leads[index] = ecg_temp[0:9000]
            print('short data after relength',index,len(ecg_leads[index]))

        elif len(ecg_leads[index]) > 9000:
            extra_index_block=[]
            if len(ecg_leads[index]) < 18000:
                ecg_leads_cut = ecg_leads[index][0:9000]

                print('cut',index,ecg_leads_cut,len(ecg_leads_cut))
                extra_index_block.append(index)
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                index_plus = index_plus+1
                extra_index_block.append(index_plus)
                ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads[index] = ecg_leads_cut
            elif len(ecg_leads[index]) ==18000:
                print(len(ecg_leads[index]))
                ecg_leads_cut = ecg_leads[index][0:9000]
                print(len(ecg_leads[index]))
                ecg_leads_extra.append(ecg_leads[index][-9000:])

                print('extra',ecg_leads_extra)
                index_plus = index_plus + 1
                extra_index_block.append(index)
                extra_index_block.append(index_plus)
                ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads[index] = ecg_leads_cut
            elif len(ecg_leads[index]) > 18000:
                iter = len(ecg_leads[index]) // 9000
                ecg_leads_cut= ecg_leads[index][:9000]
                extra_index_block.append(index)

                for i in range(1, iter):
                    start = 9000 * i
                    end = 9000 * (i + 1)
                    index_plus=index_plus+1
                    extra_index_block.append(index_plus)
                    ecg_leads_extra.append(ecg_leads[start:end])
                    ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads_extra.append(ecg_leads[index][-9000:])


                index_plus = index_plus+1
                extra_index_block.append(index_plus)
                ecg_leads[index] = ecg_leads_cut

                ecg_labels_extra.append(ecg_labels_std[index])
            extra_index.append(extra_index_block)
    ecg_labels_std = ecg_labels_std + ecg_labels_extra
    ecg_leads_std = ecg_leads + ecg_leads_extra
    print(ecg_leads_extra)
    # form the label as one-hot

    Label_set = np.zeros((len(ecg_labels_std), 4))
    #print('ppppppppppppppppp',len(ecg_leads_std))

    'change the label to a onehot'
    for i in range(len(ecg_labels_std)):
        new = np.zeros(4)
        new[int(ecg_labels_std[i])] = 1
        Label_set[i, :] = new
    return ecg_leads_std,Label_set,extra_index
def wavelet(data):

    data_denoise = []
    for i in range(len(data)):

        w = pywt.Wavelet('db8')  # Daubechies8
        maxlev = pywt.dwt_max_level(len(data[i]), w.dec_len)
        threshold = 0.08  # Threshold for filtering
        coeffs = pywt.wavedec(data[i], 'db8', level = maxlev)  # decomposition


        for j in range(1, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold* max(coeffs[j]))

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
        data_denoise.append(data2)
    return data_denoise



def butterworth(data):
    data_denoise = []
    for i in range(len(data)):
        b, a = signal.butter(8, 0.1, 'lowpass') #lowpass could also be bandpass or highpass
        data1 = signal.filtfilt(b,a,data[i])
        data_denoise.append(data1)
    return data_denoise

def compare_plot(data,datarec):
    '''compare the denoised and original ecg leads'''
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data[0:1500])
    #plt.xlabel(‘time(s)’)

    plt.title("Raw signal")
    plt.subplot(2, 1, 2)
    plt.plot(datarec[0:1500])
    plt.title("wavelet")
    plt.tight_layout()
    plt.show()
def feature_plot(feature):
    plt.figure()
    plt.plot(feature[0,1,2,:])
    plt.show()




def features_res(data):
    f_list = []
    pred_list = []
    model = tf.saved_model.load('Keras_models/new_model')
    for i in range(len(data)):
        print(i)
        lead = data[i]
        leads = leads_transfer(lead,shape=(1,50,180,1))
        predictions, features = res_feature(model, leads)
        features = features.reshape(-1, 2048)
        #pca = PCA()
        #pca.fit(features)
        #print(np.round(pca.explained_variance_ratio_, 5)[:10].sum())
        #features = pca.transform(features)[:, :10]
        features = features.reshape(-1)
        f_list.append(features)
        pred_list.append(predictions)
    f_list = np.vstack(f_list)
    pred_list = np.vstack(pred_list)
    pca = PCA()
    pca.fit(f_list)
    print(np.round(pca.explained_variance_ratio_, 40)[:40].sum())
    f_list = pca.transform(f_list)[:, :40]
    return f_list, pred_list


def leads_transfer(data,shape):#label):
    data = np.float32(data)
    #label = np.float64(label)
    #print(data)
    data = np.reshape(data, shape)

    data_t = tf.convert_to_tensor(data, np.float32)
    #label_t = tf.convert_to_tensor(label, np.float64)
    #dataset = tf.data.Dataset.zip((data_t, label_t))
    return data_t


def res_feature(model, data):
    prediction,feature4_pooled = model(data)

    return prediction.numpy(),feature4_pooled.numpy()
