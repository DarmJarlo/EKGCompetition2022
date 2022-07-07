"""
Helper Functions for train/predict
"""

from __future__ import absolute_import, division, print_function
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
import pywt
import numpy as np
from scipy.signal import medfilt
from scipy import signal
from sklearn.decomposition import PCA
import tensorflow as tf
from biosppy.signals.ecg import christov_segmenter
import neurokit2 as nk
from models.resnet import resnet_50, resnet_50_1d
from keras.callbacks import ModelCheckpoint
import config
import math
import sys

##https://blog.csdn.net/qq_36495569/article/details/104086636
#https://blog.csdn.net/qq_39594939/article/details/115697198

"""
SMOTE Algorithm to handle class imbalance
"""
def smote_algo(X, y):
    y = LabelEncoder().fit_transform(y)
    sm = SMOTE(random_state=42)
    X_synth, y_synth = sm.fit_resample(X, y)
    return X_synth, y_synth

'Denoising Methods'
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


def Locate_R(ecg,sampling_rate=300):
    '''Locate R-Peak'''
    rpeaks = christov_segmenter(signal=ecg,sampling_rate=sampling_rate)

    return rpeaks[0]


def get_model():
        # choose 1d or 2d model
    if config.Oned != True:
        model = resnet_50()
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    else:
        model = resnet_50_1d()
        model.build(input_shape=(None, 9000, 1))
    model.summary()
    if config.FurtherTrain == True:
        model = tf.saved_model.load('Keras_models/new_model')
    # print the network structure
    return model


def relength(ecg_leads,ecg_labels):
    '''make the labels as one hot and make the leads the same length with 9000
       short data : duplicate the part between the first R peak and the last R peak until 9000, if no peak detected,
       directly duplicate
       long data :sliding window with size 9000'''
    ecg_labels_std = []
    ecg_leads_extra = []
    ecg_labels_extra = []
    extra_index = []

    n = len(ecg_leads)
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
            if len(Rpeak) > 1:

                ecg_leads[index] = ecg_leads[index][Rpeak[0]:Rpeak[-1]]
                lowiter = 9000 // len(ecg_leads[index])

            else:
                lowiter = 9000 // len(ecg_leads[index])

            ecg_temp = ecg_leads[index]
            for i in range(lowiter):
                ecg_temp = np.append(ecg_temp, ecg_leads[index])
            ecg_leads[index] = ecg_temp[0:9000]

        elif len(ecg_leads[index]) > 9000:
            extra_index_block = []
            if len(ecg_leads[index]) < 18000:
                ecg_leads_cut = ecg_leads[index][0:9000]

                extra_index_block.append(index)
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                index_plus = index_plus + 1
                extra_index_block.append(index_plus)
                ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads[index] = ecg_leads_cut
            elif len(ecg_leads[index]) == 18000:
                ecg_leads_cut = ecg_leads[index][0:9000]
                ecg_leads_extra.append(ecg_leads[index][-9000:])

                index_plus = index_plus + 1
                extra_index_block.append(index)
                extra_index_block.append(index_plus)
                ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads[index] = ecg_leads_cut
            elif len(ecg_leads[index]) > 18000:
                iter = len(ecg_leads[index]) // 9000
                ecg_leads_cut = ecg_leads[index][:9000]
                extra_index_block.append(index)

                for i in range(1, iter):
                    start = 9000 * i
                    end = 9000 * (i + 1)
                    index_plus = index_plus + 1
                    extra_index_block.append(index_plus)
                    ecg_leads_extra.append(ecg_leads[index][start:end])
                    ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads_extra.append(ecg_leads[index][-9000:])

                index_plus = index_plus + 1
                extra_index_block.append(index_plus)
                ecg_leads[index] = ecg_leads_cut

                ecg_labels_extra.append(ecg_labels_std[index])
            extra_index.append(extra_index_block)
    ecg_labels_std = ecg_labels_std + ecg_labels_extra
    ecg_leads_std = ecg_leads + ecg_leads_extra

    return ecg_leads_std,ecg_labels_std,extra_index, ecg_leads_extra


def relength_leads(ecg_leads):
    ecg_leads_extra = []
    extra_index = []

    n = len(ecg_leads)
    index_plus = n - 1
    for index in range(len(ecg_leads)):
        if len(ecg_leads[index]) < 9000:
            Rpeak = Locate_R(ecg_leads[index])
            if len(Rpeak) > 1:

                ecg_leads[index] = ecg_leads[index][Rpeak[0]:Rpeak[-1]]
                lowiter = 9000 // len(ecg_leads[index])

            else:
                lowiter = 9000 // len(ecg_leads[index])

            ecg_temp = ecg_leads[index]
            for i in range(lowiter):
                ecg_temp = np.append(ecg_temp, ecg_leads[index])
            ecg_leads[index] = ecg_temp[0:9000]

        elif len(ecg_leads[index]) > 9000:
            extra_index_block = []
            if len(ecg_leads[index]) < 18000:
                ecg_leads_cut = ecg_leads[index][0:9000]

                extra_index_block.append(index)
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                index_plus = index_plus + 1
                extra_index_block.append(index_plus)
                ecg_leads[index] = ecg_leads_cut
            elif len(ecg_leads[index]) == 18000:
                ecg_leads_cut = ecg_leads[index][0:9000]
                ecg_leads_extra.append(ecg_leads[index][-9000:])

                index_plus = index_plus + 1
                extra_index_block.append(index)
                extra_index_block.append(index_plus)
                ecg_leads[index] = ecg_leads_cut
            elif len(ecg_leads[index]) > 18000:
                iter = len(ecg_leads[index]) // 9000
                ecg_leads_cut = ecg_leads[index][:9000]
                extra_index_block.append(index)

                for i in range(1, iter):
                    start = 9000 * i
                    end = 9000 * (i + 1)
                    index_plus = index_plus + 1
                    extra_index_block.append(index_plus)
                    ecg_leads_extra.append(ecg_leads[index][start:end])
                ecg_leads_extra.append(ecg_leads[index][-9000:])

                index_plus = index_plus + 1
                extra_index_block.append(index_plus)
                ecg_leads[index] = ecg_leads_cut

            extra_index.append(extra_index_block)
    ecg_leads_std = ecg_leads + ecg_leads_extra
    ecg_leads_std = np.vstack(ecg_leads_std)

    return ecg_leads_std, extra_index


def leads_transfer(data,shape):
    data = np.float32(data)
    data = np.reshape(data, shape)
    data_t = tf.convert_to_tensor(data, np.float32)
    return data_t


def res_feature(data, model):
    "get features from different levels of block"
    feature1, feature2, feature3, feature4, prediction,feature4_pooled = model(data)
    feature4_p = feature4_pooled.numpy()
    feature4_p = feature4_p.reshape((32, 64))
    kernel = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    feature4_p = signal.convolve2d(feature4_p, kernel)[::3, ::3]
    return feature1, feature2, feature3,feature4, prediction,feature4_p