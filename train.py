# -*- coding: utf-8 -*-
"""
Model 2: XGBoost Classifier

Using the hrvanalysis library given by: https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#
+ tsfel library given by: https://github.com/fraunhoferportugal/tsfel and
https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html

Features were extracted via the different methods:
(descriptions were copied from:
https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html#module-hrvanalysis.extract_features)

- time domain features
    mean_nni: The mean of RR-intervals.
    sdnn : The standard deviation of the time interval between successive normal heart beats (i.e. the RR-intervals).
    sdsd: The standard deviation of differences between adjacent RR-intervals
    rmssd: The square root of the mean of the sum of the squares of differences between adjacent NN-intervals.
    Reflects high frequency (fast or parasympathetic) influences on hrV (i.e., those influencing larger changes from
    one beat to the next).
    median_nni: Median Absolute values of the successive differences between the RR-intervals.
    nni_50: Number of interval differences of successive RR-intervals greater than 50 ms.
    pnni_50: The proportion derived by dividing nni_50 (The number of interval differences of successive RR-intervals
    greater than 50 ms) by the total number of RR-intervals.
    nni_20: Number of interval differences of successive RR-intervals greater than 20 ms.
    pnni_20: The proportion derived by dividing nni_20 (The number of interval differences of successive RR-intervals
    greater than 20 ms) by the total number of RR-intervals.
    range_nni: difference between the maximum and minimum nn_interval.
    cvsd: Coefficient of variation of successive differences equal to the rmssd divided by mean_nni.
    cvnni: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.
    mean_hr: The mean Heart Rate.
    max_hr: Max heart rate.
    min_hr: Min heart rate.
    std_hr: Standard deviation of heart rate
- frequency domain features
    total_power: Total power density spectral
    vlf: variance ( = power ) in HRV in the Very low Frequency (.003 to .04 Hz by default).
    Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic activity.
    lf: variance ( = power ) in HRV in the low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and
    parasympathetic activity, but in long-term recordings, it reflects sympathetic activity and can be reduced by the
    beta-adrenergic antagonist propanolol.
    hf: variance ( = power ) in HRV in the High Frequency (.15 to .40 Hz by default). Reflects fast changes in
    beat-to-beat variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it
    corresponds to HRV changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or
    7 breaths per minute) and decreased by anticholinergic drugs or vagal blockade.
    lf_hf_ratio: lf/hf ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal
    balance.
    lfnu: normalized lf power.
    hfnu: normalized hf power.
- geometrical features
    triangular_index: The HRV triangular index measurement is the integral of the density distribution
    (= the number of all NN-intervals) divided by the maximum of the density distribution.
    tinn: The triangular interpolation of NN-interval histogram (TINN) is the baseline width of the distribution
    measured as a base of a triangle, approximating the NN-interval distribution
- pointcare plot features
    sd1 : The standard deviation of projection of the Poincaré plot on the line perpendicular to the line of identity.
    sd2 : SD2 is defined as the standard deviation of the projection of the Poincaré plot on the line of identity (y=x)
    ratio_sd2_sd1 : Ratio between SD2 and SD1.
- csi cvi features
    csi : Cardiac Sympathetic Index.
    cvi : Cadiac Vagal Index.
    Modified_csi : Modified CSI is an alternative measure in research of seizure detection.
- sample entropy of the data

- spectral domain with provided features.json (needed for tsfel library)


model will be saved as 'model_4p_2.npy'
"""
from __future__ import absolute_import, division, print_function
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
import tensorflow as tf
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from keras.callbacks import ModelCheckpoint
import config
from prepare_data import generate_datasets
import math
import sys
sys.path.append("..")
from sklearn.model_selection import train_test_split

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads, ecg_labels, fs, ecg_names = load_references()
uniform_df = utils.uniform_length(ecg_leads, ecg_labels)
ecg_leads = uniform_df[:, :-1]
ecg_labels = uniform_df[:, -1]

X, y = utils.smote_algo(ecg_leads, ecg_labels)
y = LabelEncoder().fit_transform(y)
"""
ResNet-Training for later Feature-Extraction
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

Label_set = np.zeros((len(y), 4))

for i in range(len(y)):
    print('111111', i, y[i])
    dummy = np.zeros(4)
    dummy[int(y[i])] = 1
    Label_set[i, :] = dummy


# reshape input to be [samples, tensor shape (30 x 300)]
n = 90  # 90
m = 100  # 100
c = 1  # number of channels
X = utils.wavelet(X)

X = np.float32(X)
Label_set = np.float64(Label_set)

X_train, X_val, y_train, y_val = train_test_split(ecg_leads, ecg_labels, test_size=0.2, random_state=42)
X_train = np.float32(X_train)
X_val = np.float32(X_val)
y_train = np.float64(y_train)
y_val = np.float64(y_val)

X_train = np.reshape(X_train, (len(X_train), n, m, c))
X_val = np.reshape(X_val, (len(X_val), n, m, c))
image_size = (n, m, c)

# create model
model = resnet_50()
model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
model.summary()# print the network structure

# define loss and optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy()

alpha = 0.01
accu = 0
#optimizer = tf.keras.optimizers.Adadelta(learning_rate=alpha)
optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=0.001,
    initial_accumulator_value=0.1,
    epsilon=1e-07,
    name="Adagrad"
)
# optimizer = tf.keras.optimizers.Adadelta(learning_rate=1) #10 iteration 0.81  every iteration has better result. BUt maybe overfitting
# optimizer = tf.keras.optimizers.Adagrad(
#    learning_rate=0.001,
#    initial_accumulator_value=0.1,
#    epsilon=1e-07,
#    name="Adagrad"
# )
# optimizer= tf.keras.optimizers.Adam(learning_rate=0.1)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')


@tf.function
def train_step(images, labels, epoch):
    # print('zzzzzzzzzzzzzz')
    with tf.GradientTape() as tape:
        predictions, feature4_pooled = model(images, training=True)  # dont forget here we are inputing a whole batch
        print('oooooooooooo', epoch, feature4_pooled)
        # predictions = model(images, training=True)
        # predictions = predictions.numpy()
        # predictions = predictions/np.sum(predictions)
        print(predictions)

        # print(labels)
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    # print("gradients",gradients)

    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    return gradients


@tf.function
def valid_step(images, labels):
    predictions, feature4_pooled = model(images, training=False)
    # predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)


X_train = tf.data.Dataset.from_tensor_slices(X_train)
Y_train = tf.data.Dataset.from_tensor_slices(y_train)
X_val = tf.data.Dataset.from_tensor_slices(X_val)
Y_val = tf.data.Dataset.from_tensor_slices(y_val)
train_dataset = tf.data.Dataset.zip((X_train, Y_train))
valid_dataset = tf.data.Dataset.zip((X_val, Y_val))

train_count = len(train_dataset)
print("traincount", np.array(train_count).shape)

train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
# print('ddddddddddddddddd',train_dataset)
valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
# start training
for epoch in range(config.EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    if epoch < 6:
        alpha = epoch * 0.2 + 0.2
    elif epoch > 15 and accu < 0.85:
        alpha = 0.2 + (epoch - 16 * 0.2)
    else:
        alpha = alpha / 5
    print("alpha", alpha)
    # optimizer_1 = tf.keras.optimizers.Adam(learning_rate=alpha)
    step = 0
    for images, labels in train_dataset:
        # print("images",images.shape,labels.shape)
        step += 1
        # gradients=train_step(images, labels)
        train_step(images, labels, epoch)
        print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                 config.EPOCHS,
                                                                                 step,
                                                                                 math.ceil(
                                                                                     train_count / config.BATCH_SIZE),
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result()))
        # print(gradients)

    for valid_images, valid_labels in valid_dataset:
        valid_step(valid_images, valid_labels)

    print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
          "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                              config.EPOCHS,
                                                              train_loss.result(),
                                                              train_accuracy.result(),
                                                              valid_loss.result(),
                                                              valid_accuracy.result()))

    # dont forget to add valid dataset back to train set
    accu = valid_accuracy.result()
'''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                               monitor='val_accuracy',
                               save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
model.save('resnet50')  # save trained ResNet for feature extraction

features_res, predictions_res = utils.features_res(X)

detectors = Detectors(fs)
cfg = tsfel.get_features_by_domain(domain='spectral', json_path='features.json')

feature_vector = np.array([])  # create empty arrays for features and targets
targets = np.array([])

for idx, ecg_lead in enumerate(ecg_leads):
    spectral_features = tsfel.time_series_features_extractor(cfg, ecg_lead, fs=fs)
    corr_features = tsfel.correlated_features(spectral_features)
    spectral_features.drop(corr_features, axis=1, inplace=True)
    spectral_features = spectral_features.to_numpy()    # extracting spectral features

    rr_intervals = detectors.two_average_detector(ecg_lead)

    if len(rr_intervals) == 1:
        rr_intervals = np.abs(rr_intervals)
        arti_rr_1 = rr_intervals * random.random()
        rr_intervals = np.append(rr_intervals, [arti_rr_1])     # append rr_intervals to have at least two values

    rr_intervals_ms = np.diff(rr_intervals) / fs * 1000  # Umwandlung in ms
    rr_intervals_ms = [abs(number) for number in rr_intervals_ms]  # make rr_intervals positive

    rr_without_outliers = remove_outliers(rr_intervals_ms, low_rri=300, high_rri=2000)  # preprocessing
    rr_intervals_list = interpolate_nan_values(rr_without_outliers, interpolation_method='linear')

    rr_intervals_list = [x for x in rr_intervals_list if str(x) != 'nan']  # remove nan values

    if len(rr_intervals_list) <= 2:       # frequency domain needs at least 3 values
        mean_rr = np.nanmean(rr_intervals_list)
        rr_intervals_list = np.nan_to_num(rr_intervals, nan=mean_rr)
        arti_rr_1 = rr_intervals_list[0] * random.random()
        arti_rr_2 = rr_intervals_list[0] * random.random()
        rr_intervals_list = np.append(rr_intervals_list, arti_rr_1)
        rr_intervals_list = np.append(rr_intervals_list, arti_rr_2)

    dict_time_domain = hrv.get_time_domain_features(rr_intervals_list)  # feature extraction via hrv
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

    feature_vector = np.append(feature_vector, values_time)
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

df = df.assign(Labels=ecg_labels)

features_xgb = df.to_numpy()
labels_xgb = y
#features_xgb = df[:, :-1]
#labels_xgb = df[:, -1]



xgb = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6, min_child_weight=0, gamma=0,
                          subsample=0.55, colsample_bytree=0.75, bjective='multi:softmax',
                          nthread=4, scale_pos_weight=1, seed=42)
xgb.fit(features_xgb, labels_xgb)                # fit XGBoost Classifier


if os.path.exists("model_4p_3.npy"):
    os.remove("model_4p_3.npy")
with open('model_4p_3.npy', 'wb') as f:
    pickle.dump(xgb, f)          # save model

print('Training is done')

