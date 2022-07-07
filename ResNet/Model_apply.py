from tensorflow.keras.applications import ResNet50
import tensorflow.keras as keras
import tensorflow as tf
from scipy import signal
from keras import Model
import sys
sys.path.append("..")
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from utils import wavelet, compare_plot, normalize,median_filter, butterworth,relength,feature_plot,Locate_R
import config
from models.resnet import resnet_50, resnet_50_1d
from wettbewerb import load_references

"this file is about applying and evaluate the model"
def leads_transfer(data,shape,label):
    data = np.float32(data)
    label = np.float64(label)
    data = np.reshape(data, shape)
    data_t = tf.convert_to_tensor(data, np.float32)
    label_t = tf.convert_to_tensor(label, np.float64)
    #dataset = tf.data.Dataset.zip((data_t, label_t))
    return data_t

def res_feature(data):
    "get features from different levels of block"
    # choose 1d or 2d model
    if config.Oned != True:
        model = resnet_50()
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    else:
        model = resnet_50_1d()
        model.build(input_shape=(None, 9000, 1))
    model = tf.saved_model.load('Keras_models/new_model')
    feature1, feature2, feature3, feature4, prediction,feature4_pooled = model(data)
    feature4_p = feature4_pooled.numpy()
    feature4_p = feature4_p.reshape((32, 64))
    kernel = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    feature4_p = signal.convolve2d(feature4_p, kernel)[::3, ::3]
    return feature1, feature2, feature3,feature4, prediction,feature4_p



if __name__ == '__main__':
    ecg_leads, ecg_labels, fs, ecg_names = load_references()
    print('ddddddddddd',len(ecg_leads[10]))
    #compare_plot(ecg_leads[1217], ecg_leads[1495])
    first = 0
    end =  -1
    leads, labels ,extra_index = relength(ecg_leads[first:end], ecg_labels[first:end])
    ind =[]
    fal =[]
    for i in range(7000):
        ind.append(len(leads[i]))
        print('...........',ind,i)
        if ind[i] == 0:
            fal.append(i)
            print(fal)
    print(extra_index)
'''
    predictions = []
    print('index',extra_index,len(ecg_leads[0]))

    features = []


    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    last_accuracy = 1
    y_pred = []
    label_int =[]
    false_index = []
    false_length = []
    false_pred = []
    if config.Oned != True:
        shape =(1,90,100,1)
    else:
        shape = (1,9000,1)
    for index in range(first,end):
        lead = leads_transfer(leads[index-first], shape, labels[index-first])
        feature1,feature2,feature3,feature4, prediction,feature4_p = res_feature(lead)
        prediction = prediction.numpy()
        label_predi=np.argmax(prediction)
        label_int.append(np.argmax(labels[index-first]))
        y_pred.append(label_predi)
        print(index,prediction)
        #feature4 = feature4.numpy()
        #feature3 = feature3.numpy()
        #feature2 = feature2.numpy()
        #feature1 = feature1.numpy()
        #print('feature',feature4_p.shape)

        test_accuracy(labels[index-first], prediction)
        if test_accuracy.result() < last_accuracy:
            false_index.append(index)
            false_length.append(len(ecg_leads[index]))
            false_pred.append([label_predi,ecg_labels[index]])

        last_accuracy = test_accuracy.result()

        #print(feature4_p.shape)
        #labels=np.array(labels)
        #y_pred = np.array(y_pred)
        print(label_int)
        print(y_pred)
        print("Epoch: {},  accuracy: {:.5f}".format(index, test_accuracy.result()))
    print('Accuracy:', metrics.accuracy_score(label_int, y_pred))
    print('Precision:', metrics.precision_score(label_int, y_pred, average=None))
    print('Recall:', metrics.recall_score(label_int, y_pred, average=None))
    print('F1:', metrics.f1_score(label_int,y_pred, average=None))
    print('false_index',false_index)
    print('false_length', false_length)
    print('false_pred:[pred,true_value]',false_pred)
'''
