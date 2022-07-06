
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
#https://neuropsychology.github.io/NeuroKit/examples/ecg_delineate/ecg_delineate.html

def PCA_single_reduce(features):
    '''
    input should be a list
    two ways to use pca:
    first one is initially to calculate which index of feature is important and directly use these pretrained index
     second one is to train it again with the input test data and select the important features'''

    print('9888888',features.shape)
    c,r = features.shape
    features=np.array(features)
    print(features)
    pca = PCA(n_components=c*r)

    newX = pca.fit_transform(features)

    m = pca.explained_variance_ratio_
    print(m)
    list = np.arange(r)
    m=np.vstack(m,list)
    print(m)
    m = m(m[0,:].argsort())
    sum=0
    index = []

    for i in range(r):
        index.append(m[1][i])
        sum = m[0][i]+sum
        if sum>0.9:
            break


    return index
    #return firsthalfindex
def Locate_R(ecg,sampling_rate=300):
    rpeaks = christov_segmenter(signal=ecg,sampling_rate=sampling_rate)
    #_, rpeaks = nk.ecg_peaks(ecg, sampling_rate=300)
    #_, waves_peak = nk.ecg_delineate(ecg_data, rpeaks, sampling_rate=300, method="peak")
    #plot = nk.events_plot(waves_peak['ECG_S_Peaks'][:3])
    print(rpeaks)

    # Preprocess ECG signal
    #signals, info = nk.ecg_process(ecg, sampling_rate=300)

    # Visualize
    #nk.ecg_plot(signals)
    dex= []
    #for index in range(len(ecg)):
    #    if signals["ECG_R_Peaks"][index] == 1:
    #        dex.append(index)
    #for index in range(len(ecg)):
     #  if rpeaks['ECG_R_Peaks'][index] == 1:
      #     dex.append(index)
        # if we cannot find the peak by this approach ,we may do the gradient search
        # in 1 2 3 4 order or in 5 4 3 2 1 order to find the first peak and the last peak
    #return rpeaks['ECG_R_Peaks']
    return rpeaks[0]
def relength(ecg_leads,ecg_labels):
    '''make the labels as one hot and make the leads the same length with 9000'''
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
            print('iiiiiiiiii')
            Rpeak = Locate_R(ecg_leads[index])
            if len(Rpeak)>1:
                print('qqqqqqqqqqqqq',len(Rpeak),index)
                ecg_leads[index]=ecg_leads[index][Rpeak[0]:Rpeak[-1]]
                print("rpeaks",Rpeak[0],Rpeak[-1],len(ecg_leads[index]))
                lowiter = 9000 // len(ecg_leads[index])
                print(lowiter)
            else:
                lowiter = 9000 // len(ecg_leads[index])
                print(lowiter)


            ecg_temp=ecg_leads[index]
            for i in range(lowiter):
                print(ecg_leads[index].shape)
                ecg_temp = np.append(ecg_temp, ecg_leads[index])
                print('dadadad', ecg_leads[index].shape)
            ecg_leads[index] = ecg_temp[0:9000]
            print('short data after relength',index,len(ecg_leads[index]))
        elif len(ecg_leads[index]) > 9000:
            extra_index_block=[]
            if len(ecg_leads[index] <= 18000):
                ecg_leads[index] = ecg_leads[index][0:9000]
                extra_index_block.append(index)
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                index_plus = index_plus+1
                extra_index_block.append(index_plus)
                ecg_labels_extra.append(ecg_labels_std[index])
            elif len(ecg_leads[index] > 18000):
                iter = len(ecg_leads[index]) // 9000
                ecg_leads[index] = ecg_leads[index][:9000]
                extra_index_block.append(index)
                index_plus = index_plus+1
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

                ecg_labels_extra.append(ecg_labels_std[index])
            extra_index.append(extra_index_block)
    ecg_labels_std = ecg_labels_std + ecg_labels_extra
    ecg_leads_std = ecg_leads + ecg_leads_extra
    print('jjjjjjj',extra_index)
    print(';;;;;;;;;;',len(ecg_leads_extra))
    print(ecg_labels_extra)
    # form the label as one-hot

    Label_set = np.zeros((len(ecg_labels_std), 4))
    print('ppppppppppppppppp',len(ecg_leads_std))
    for i in range(len(ecg_labels_std)):
        print('111111', i, ecg_labels_std[i])
        dummy = np.zeros(4)
        dummy[int(ecg_labels_std[i])] = 1
        Label_set[i, :] = dummy
    return ecg_leads_std,Label_set,extra_index
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

        data3 = medfilt(data1,3)

        data_denoise.append(data3)
    return data_denoise
#def s_peak_medianfilter(data,s_peak):


def butterworth(data):
    data_denoise = []
    for i in range(len(data)):
        b, a = signal.butter(8,0.1, 'lowpass') #lowpass could also be bandpass or highpass
        data1 = signal.filtfilt(b,a,data[i])
        data_denoise.append(data1)
    return data_denoise




def compare_plot(data,datarec):
    '''compare the denoised and original ecg leads'''
    mintime = 0
    maxtime = mintime + len(data) + 1
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(data[0:1500])
    #plt.xlabel(‘time(s)’)

    plt.title("Raw signal")
    plt.subplot(2, 1, 2)
    plt.plot(datarec[0:1500])
   # plt.xlabel(‘time(s)’)

    plt.title("wavelet")

    plt.tight_layout()
    plt.show()
def feature_plot(feature):
    plt.figure()

    plt.plot(feature[0,1,2,:])
    plt.show()
#def extra_index_pred(ecg_leads,extra_index)
def smote_NN(data,label):
    #in fact a kmeanssmote
    y = LabelEncoder().fit_transform(y)
    sm = KMeansSMOTE(random_state=42)
    X_synth, y_synth = sm.fit_resample(X, y)
    return X_synth, y_synth

def smote_algo(X, y):
    y = LabelEncoder().fit_transform(y)
    sm = SMOTE(random_state=42)
    X_synth, y_synth = sm.fit_resample(X, y)
    return X_synth, y_synth