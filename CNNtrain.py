from wettbewerb import load_references


'''https://www.youtube.com/watch?v=WvoLTXIjBYU
https://keras.io/api/layers/pooling_layers/max_pooling2d/
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout,Activation

import tensorflow as tf
model = Sequential()
model.add( Conv2D(32, kernel_size=3, input_shape=input.shape[1:]))
model.add( Activation(act_func))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, input_shape=input.shape[1:]))
model.add(Activation(act_func))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(4, activation='softmax'))

'''
ecg_leads,ecg_labels,fs,ecg_names = load_references()
print(ecg_labels)
for i in range(len(ecg_labels)):
    ecg_labels[i] = 0 if ecg_labels[i] == 'N'else 1 if ecg_labels[i] == 'A'else 2 if ecg_labels[i] == 'O'else 3 if ecg_labels[i]== '~' else ecg_labels
    #normal 0 afib 1 othre 2 noise 3
print(ecg_labels)

y_train = ecg_labels
x_train = ecg_leads
len_leads = []
for i in range(len(x_train)):
    len_leads.append(len(x_train[i]))
    print(len_leads)




