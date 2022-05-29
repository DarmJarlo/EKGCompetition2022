import numpy
from wettbewerb import load_references
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from sklearn.metrics import accuracy_score
from keras import backend as K
import sys
#adopted from https://github.com/ismorphism/DeepECG/blob/master/CNN_ECG.py
#K.image_data_format('tf')  # For problems with ordering

number_of_classes = 2


def change(x):
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)



# Loading of .mat files from training directory. Only 9000 time steps from every ECG file is loaded

#for i in range(len(mats)):
 #   if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
  #      target_train[i] = 0
   # elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
    #    target_train[i] = 1
    #elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'O':
     #   target_train[i] = 2
    #else:
     #   target_train[i] = 3
ecg_leads,ecg_labels,fs,ecg_names = load_references()
ecg_labels_std =[]
ecg_leads_std=[]
for index in range(len(ecg_labels)):
    if len(ecg_leads[index]) == 9000 :

        if ecg_labels[index] == 'N' or ecg_labels[index] == 'A':
            ecg_leads_std.append(ecg_leads[index])
            if ecg_labels[index] == 'N':
                ecg_labels_std.append(0)
            else:
                ecg_labels_std.append(1)
Label_set = np.zeros((len(ecg_labels_std), 2))
ecg_leads_std = np.array(ecg_leads_std)
for i in range(len(ecg_labels_std)):
    dummy = np.zeros(2)
    dummy[int(ecg_labels_std[i])] = 1
    Label_set[i, :] = dummy


# X = np.abs(numpy.fft.fft(X)) #some stuff

# Normalization part
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)

mats = len(ecg_leads_std)
print(ecg_leads_std.shape)
#print(len(mats))
X=ecg_leads_std

train_len = 0.8  # Choice of training size
var=int(train_len * (mats))
X_train = X[:367]
Y_train = Label_set[:367]

#Y_train = Label_set[:int(train_len * (mats))]
print(Y_train)
#X_val = X[int(train_len * (mats)):]
#Y_val = Label_set[int(train_len * (mats)):]
X_val = X[2367:]
Y_val = Label_set[2367:]
# reshape input to be [samples, tensor shape (30 x 300)]
n = 20
m = 450
c = 1  # number of channels

X_train = numpy.reshape(X_train, (X_train.shape[0], n, m, c))
X_val = numpy.reshape(X_val, (X_val.shape[0], n, m, c))
image_size = (n, m, c)

# create and fit the CNN network

batch_size = 32
model = Sequential()
# model.load_weights('my_model_weights.h5')
# 64 conv
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=image_size, padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 128 conv
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# #256 conv
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# #512 conv
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(Conv2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Dense part
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Callbacks and accuracy calculation
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy',
                               save_weights_only=False, period=1, verbose=1, save_best_only=False)
model.fit(X_train, Y_train, epochs=2, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=2, shuffle=False,
          callbacks=[checkpointer])
model.save('Keras_models/my_model_' + '.h5')
predictions = model.predict(X_val)
score = accuracy_score(change(Y_val), change(predictions))
print(score)
# Data[i - starti, j - starti] = str(format(score, '.5f'))
# Output = pd.DataFrame(Data)
# name = str(batch_size) + '.csv'
# Output.to_csv(path_or_buf='Keras_models/' + name, index=None, header=None)