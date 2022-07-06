from tensorflow.keras.applications import ResNet50
import tensorflow.keras as keras
import tensorflow as tf
from scipy import signal
from keras import Model
import sys
sys.path.append("..")
import numpy as np

import matplotlib.pyplot as plt
from wettbewerb import load_references
from tensorflow.keras.applications import ResNet50
from Denoise import wavelet, compare_plot, normalize,median_filter, butterworth,relength,feature_plot, PCA_single_reduce,Locate_R
import config
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from wettbewerb import load_references

def leads_transfer(data,shape,label):
    data = np.float32(data)
    label = np.float64(label)
    print(data)
    data = np.reshape(data, shape)

    data_t = tf.convert_to_tensor(data, np.float32)
    label_t = tf.convert_to_tensor(label, np.float64)
    #dataset = tf.data.Dataset.zip((data_t, label_t))
    return data_t


def res_feature(data):
    def get_model():
    #model = ResNet50()
        model = resnet_50()
        if config.model == "resnet18":
            model = resnet_18()
        if config.model == "resnet34":
            model = resnet_34()
        if config.model == "resnet101":
            model = resnet_101()
        if config.model == "resnet152":
            model = resnet_152()
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
        model.summary()# print the network structure
        return model

##model=get_model()
#Res = ResNet50( include_top=True, weights=config.save_model_dir,input_shape=(50,180,1))
#model= tf.saved_model.load('saved_model/model')
#model.load_weights(config.save_model_dir)
#print(model.summary)
#no_top = Model(inputs = model.input,outputs = model.get_layer('avgpool').output)



    model = tf.saved_model.load('Keras_models/new_model')
    feature1, feature2, feature3, feature4, prediction,feature4_pooled = model(data)
    # print(prediction)
    # print(feature4)
    return feature1, feature2, feature3,feature4, prediction,feature4_pooled
#x = model.layers[-1].output
#x = Dense(256)(x)
#predictions = Dense(15, activation = "softmax")(x)
#model = Model(inputs = model.input, outputs = x)


if __name__ == '__main__':
    ecg_leads, ecg_labels, fs, ecg_names = load_references()
    #for i in range(len(ecg_leads)):
     #   r_peaks = Locate_R(ecg_leads[i],sampling_rate=300)
      #  print(r_peaks)
    compare_plot(ecg_leads[1193], ecg_leads[505])

    leads, labels ,extra_index = relength(ecg_leads, ecg_labels)
    predictions = []
    print('index',extra_index,len(ecg_leads[0]))

    '''for i in range(len(labels)):
        print('2222', i, labels[i])
        dummy = np.zeros(4)
        dummy[int(labels[i])] = 1
        Label_set[i, :] = dummy
    Y_val = tf.data.Dataset.from_tensor_slices(Label_set)
    '''

    features = []
    i=4999

    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    for index in range(5300,5301):
        lead = leads_transfer(leads[index], (1, 90, 100, 1), labels[index])
        feature1,feature2,feature3,feature4, prediction,feature4_pooled = res_feature(lead)
        prediction = prediction.numpy()
        i=i+1
        print(i,prediction)
        feature4 = feature4.numpy()
        feature3 = feature3.numpy()
        feature2 = feature2.numpy()
        feature1 = feature1.numpy()
        #a,b,c,d=feature4.shape
        feature4_p= feature4_pooled.numpy()
        feature4_p=feature4_p.reshape((32,64))
        print(feature4_p.shape)
        kernel = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
        feature4_p = signal.convolve2d(feature4_p, kernel)[ ::3,::3]
#add the padding mode tomorrow
        test_accuracy(labels[index], prediction)
        print(feature4_p.shape)
        features.append(feature4)
        #print(feature4)
        #print(feature4.shape)
        #print(feature3)
        #print(feature3.shape)
        #print(feature2)
        #print(feature2.shape)
        #print(feature1.shape)
        #feature_plot(feature4)
        print("Epoch: {},  accuracy: {:.5f}".format(index + 1, test_accuracy.result()))
    print(features)
    #index = PCA_single_reduce(features)
    #valid_accur        print(feature3.shape)acy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')
    #valid_accuracy(Y_val, predictions)
    #print(valid_accuracy.result)