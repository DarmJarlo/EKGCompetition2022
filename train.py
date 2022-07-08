# -*- coding: utf-8 -*-
"""
Model 4: XGBoost Classifier & ResNet50 - Ensemble

Using the hrvanalysis library given by: https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#
+ tsfel library given by: https://github.com/fraunhoferportugal/tsfel and
https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html

XGBClassifier will be saved as 'model_4p_4.npy'
ResNet will be saved in Keras_models folder
"""

import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references
import hrvanalysis as hrv
from hrvanalysis.preprocessing import remove_outliers, interpolate_nan_values
import random
import pandas as pd
import tsfel
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import utils
import config
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads, ecg_labels, fs, ecg_names = load_references()

print('start uniform length')
ecg_leads_std, ecg_labels_std, extra_index, ecg_lead_extra = utils.relength(ecg_leads, ecg_labels)
print('uniform length done')

X = np.vstack(ecg_leads_std)
y = np.vstack(ecg_labels_std)

y = LabelEncoder().fit_transform(y)
sm = SMOTE(random_state=42)     # handle class_imbalance using oversampling via SMOTE
X, y = sm.fit_resample(X, y)
print('SMOTE oversampling done')


def train_resnet(X, y):
    print('Start Training of Resnet')
    Label_set_train = np.zeros((len(y), 4))

    for i in range(len(y)):  # prepare labels for resnet-training
        dummy_test = np.zeros(4)
        dummy_test[int(y[i])] = 1
        Label_set_train[i, :] = dummy_test

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    'Denoising: wavelet and butterworth low-pass'
    #X = wavelet(X)
    #X = butterworth(X)

    'Preparation of Train data '
    X_train = np.float32(X)
    y_train = np.float64(Label_set_train)
    print(X_train.shape[0])

    if config.Oned != True:
        X_train = np.reshape(X_train, (X_train.shape[0], config.image_height, config.image_width, config.channels))
        image_size = (config.image_height, config.image_width, config.channels)
    else:
        X_train = np.reshape(X_train, (X_train.shape[0], 9000, 1))
        image_size = (9000, 1)

    'transform to tensor'
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    Y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((X_train, Y_train))

    train_count = len(train_dataset)
    print("traincount", np.array(train_count).shape)

    'batch shuffle'
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)

    # create model
    model = utils.get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    alpha = 0.01
    accu = 0
    #optimizer_1 = tf.keras.optimizers.Adadelta(learning_rate=alpha)

    optimizer_1 = tf.keras.optimizers.Adagrad(
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )
    #optimizer_1= tf.keras.optimizers.Adam(learning_rate=0.2)
    train_loss = tf.keras.metrics.Mean(name='train_loss')  # this mean only calculate the  mean of the loss
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


    @tf.function
    def train_step(images, labels, epoch):
        with tf.GradientTape() as tape:
            feature1, feature2, feature3, feature4, predictions, feature4_pooled = model(images,
                                                                                         training=True)  # dont forget here we are inputing a whole batch
            print('oooooooooooo', epoch, feature4_pooled)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        return gradients

    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()

        'adaptive learning rate for adadelta'

        if epoch < 6:
            alpha = epoch * 0.2 + 0.2
        elif epoch > 15 and accu < 0.85:
            alpha = 0.2 + (epoch - 16 * 0.2)
        else:
            alpha = alpha / 5
        print("alpha", alpha)

        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels, epoch)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     config.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(
                                                                                         train_count / config.BATCH_SIZE),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

        tf.saved_model.save(model, 'Keras_models/new_model')
    '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                   monitor='val_accuracy',
                                   save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
    tf.saved_model.save(model, 'Keras_models/new_model')


def train_xgb(X, y):
    print('Start Training of XGB')
    detectors = Detectors(fs)
    cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

    feature_vector = np.array([])  # create empty arrays for extracted features
    for idx in range(len(X)):
        ecg_lead = X[idx]
        spectral_features = tsfel.time_series_features_extractor(cfg, ecg_lead, fs=fs)
        corr_features = tsfel.correlated_features(spectral_features)
        spectral_features.drop(corr_features, axis=1, inplace=True)
        spectral_features = spectral_features.to_numpy()  # extract spectral features

        rr_intervals = detectors.two_average_detector(ecg_lead)  # get rr-intervals
        if len(rr_intervals) == 1:                               # and prepare them for feature extraction
            rr_intervals = np.abs(rr_intervals)
            arti_rr_1 = rr_intervals * random.random()
            rr_intervals = np.append(rr_intervals, [arti_rr_1])

        rr_intervals_ms = np.diff(rr_intervals) / fs * 1000  # Umwandlung in ms
        rr_intervals_ms = [abs(number) for number in rr_intervals_ms]

        rr_without_outliers = remove_outliers(rr_intervals_ms, low_rri=300, high_rri=2000)
        rr_intervals_list = interpolate_nan_values(rr_without_outliers, interpolation_method='linear')

        rr_intervals_list = [x for x in rr_intervals_list if str(x) != 'nan']  # remove nan values

        if len(rr_intervals_list) <= 2:  # frequency domain needs at least 3 values
            mean_rr = np.nanmean(rr_intervals_list)
            rr_intervals_list = np.nan_to_num(rr_intervals, nan=mean_rr)
            arti_rr_1 = rr_intervals_list[0] * random.random()
            arti_rr_2 = rr_intervals_list[0] * random.random()
            rr_intervals_list = np.append(rr_intervals_list, arti_rr_1)
            rr_intervals_list = np.append(rr_intervals_list, arti_rr_2)

        dict_time_domain = hrv.get_time_domain_features(rr_intervals_list)  # extract features from hrv-analysis
        dict_geometrical_features = hrv.get_geometrical_features(rr_intervals_list)
        dict_pointcare = hrv.get_poincare_plot_features(rr_intervals_list)
        dict_csi_csv = hrv.get_csi_cvi_features(rr_intervals_list)
        dict_entropy = hrv.get_sampen(rr_intervals_list)
        dict_frequency_domain = hrv.get_frequency_domain_features(rr_intervals_list)

        values_time = list(dict_time_domain.values())
        values_frequency = list(dict_frequency_domain.values())
        values_geometrical = list(dict_geometrical_features.values())
        values_pointcare = list(dict_pointcare.values())
        values_entropy = list(dict_entropy.values())
        values_csicsv = list(dict_csi_csv.values())

        feature_vector = np.append(feature_vector, values_time)  # create feature vector
        feature_vector = np.append(feature_vector, values_frequency)
        feature_vector = np.append(feature_vector, values_geometrical)
        feature_vector = np.append(feature_vector, values_pointcare)
        feature_vector = np.append(feature_vector, values_entropy)
        feature_vector = np.append(feature_vector, values_csicsv)
        feature_vector = np.append(feature_vector, spectral_features)

        if (idx % 100) == 0:
            print(str(idx) + "\t EKG Signale wurden verarbeitet.")

    feature_vector = np.reshape(feature_vector, (int(len(feature_vector) / 57), 57))  # reshape fv

    feature_names = ['mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
                     'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'total_power', 'vlf', 'lf',
                     'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'triangular_index', 'tinn', 'sd1', 'sd2', 'ratio_sd2_sd1',
                     'csi', 'cvi', 'Modified_csi', 'sampen', 'Fundamental freq', 'Human energy range',
                     'Max power spectrum', 'Max Frequency', 'Median Frequency', 'Power bandwith',
                     'Spectral centroid', 'Spectral decrease', 'Spectral entropy', 'Spectral kurtosis',
                     'Spectral positive turning points', 'Spectral roll-off', 'Spectral roll-on', 'Spectral skewness',
                     'Spectral spread', 'Spectral variation', 'Wavelet abs mean 1', 'Wavelet abs mean 2',
                     'Wavelet abs mean 3', 'Wavelet abs mean 4', 'Wavelet abs mean 5', 'Wavelet abs mean 6',
                     'Wavelet abs mean 7', 'Wavelet abs mean 8', 'Wavelet abs mean 9']

    index = np.arange(len(feature_vector))
    df = pd.DataFrame(data=feature_vector, index=index, columns=feature_names)
    df = df.drop(columns=['tinn'])  # column has None-values

    df = df.replace([np.inf, -np.inf], np.nan)   # Replace other invalid values
    column_means = df.mean()
    df = df.fillna(column_means)

    df = df.assign(Labels=y)

    df = df.to_numpy()
    X = df[:, :-1]
    y = df[:, -1]

    xgb = XGBClassifier(learning_rate=0.1, n_estimators=1500, max_depth=6, min_child_weight=0, gamma=0,
                          subsample=0.55, colsample_bytree=0.75, bjective='multi:softmax',
                          nthread=4, scale_pos_weight=1, seed=42)
    xgb.fit(X, y)   # fit XGBoost Classifier

    if os.path.exists("model_4p_4.npy"):
        os.remove("model_4p_4.npy")
    with open('model_4p_4.npy', 'wb') as f:
        pickle.dump(xgb, f)          # save model

    print('Training is done')


train_resnet(X, y)

train_xgb(X, y)
