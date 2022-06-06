
import matplotlib.pyplot as plt
import pywt
import numpy as np
import os
##https://blog.csdn.net/qq_36495569/article/details/104086636
def wavelet(data):

    data_denoise = []
    for i in range(len(data)):

        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data[i]), w.dec_len)
        print("maximum level is " + str(maxlev))
        threshold = 0.04  # Threshold for filtering
        coeffs = pywt.wavedec(data[i], 'db8', level = maxlev)  # 将信号进行小波分解

        plt.figure()
        for j in range(1, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))  # 将噪声滤波

        datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
        data_denoise.append(datarec)

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

