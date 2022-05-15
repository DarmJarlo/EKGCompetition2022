from wettbewerb import load_references
import numpy as np
from matplotlib import pyplot as plt
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
max_len = max(len_leads)
print(max_len)
count = 0
x_normal = []
index_normal = []
for index in range(len(x_train)):
    if ecg_labels[index] == 0 and len_leads[index] == 9000:#here this 9000 must be replaced by the number with most valid data later
        count = count +1
        x_normal.append(ecg_leads[index])
        index_normal.append(index)

num_normal = len(x_normal)

#for index in range(len(ecg_normal)):
 #   x_normal.append(ecg_leads[ecg_normal[index]])
  #  num_normal.append(len(ecg_leads[_normal[index]]))

print(index_normal)
print(num_normal)

template_normal = np.sum(x_normal,axis=0).tolist()
template_normal = [factor/num_normal for factor in template_normal]
x= range(9000)
fig= plt.figure(figsize=(15,5))
plt.plot(x,template_normal,'bo-')

plt.show()

#print(template_normal)

#idea is like because there are different sizes of data.some are 3000 some are even 18000, and because most are with 9000 ,so
# we use the normal data with length of 9000 to get a template(sum it up) and use this template to pad
#  the other data with short length and for the long data, we divide it
# and padd the short part
#but maybe we should first ignore all the noise data