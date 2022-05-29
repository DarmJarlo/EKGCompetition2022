from wettbewerb import load_references
import numpy as np
from matplotlib import pyplot as plt
import Locate
import neurokit2 as nk
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from keras.models import Sequential
import tensorflow.keras as keras
from keras import Model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout,Activation
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
tf.disable_eager_execution()
from cnn.Network import Network
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
#plt.rcParams['figure.figsize'] = [8, 5]
ecg_leads,ecg_labels,fs,ecg_names = load_references()
#_, rpeaks = nk.ecg_peaks(ecg_leads[0], sampling_rate=30)
# Visualize R-peaks in ECG signal
#plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_leads[0])
#plt.show()

ecg_leads_std =[]
ecg_labels_std = []

for index in range(len(ecg_labels)):
    if len(ecg_leads[index]) == 9000 :

        if ecg_labels[index] == 'N' or ecg_labels[index] == 'A':
            ecg_leads_std.append(ecg_leads[index])
            if ecg_labels[index] == 'N':
                ecg_labels_std.append(0)
            else:
                ecg_labels_std.append(1)

print(len(ecg_leads_std))

#Model = Network()
#Model = Model.load_model("model")
ecg_leads_std = np.array(ecg_leads_std)
print(ecg_leads_std.shape)

ecg_labels_std = np.array(ecg_labels_std)
x = np.reshape(ecg_leads_std, (ecg_leads_std.shape[0], 20, 450, 1))
y = np.reshape(ecg_labels_std, (ecg_labels_std.shape[0],1))
model = Sequential()
#model.load_weights('my_model_weights.h5')
#64 conv
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(20,450,1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#128 conv
model.add(Conv2D(128, (3, 3), activation='relu', padding='same' ))
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

#Dense part
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))
#model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#              loss=keras.losses.BinaryCrossentropy(),
 #             metrics=[keras.metrics.BinaryAccuracy(),
  #                     keras.metrics.FalseNegatives()])

#model.fit(x,y,batch_size=32)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_loss', save_weights_only=False, period=1, verbose=1, save_best_only=False)
model.fit(x, y, epochs=250, batch_size=32, verbose=2, shuffle=False, callbacks=[checkpointer])
model.save('Keras_models/my_model_' + str(i) + '_' + str(j) + '_' + str() + '.h5')
#predictions = model.predict(X_val)
#score = accuracy_score(change(Y_val), change(predictions))
#print(score)











#ecg_leads_std = np.array(ecg_leads_std)

#ecg_labels_std = np.array(ecg_labels_std)
#print(ecg_labels_std.shape)
#for i in range(len(ecg_leads_std)):

#ecg_leads_std= tf.convert_to_tensor(ecg_leads_std,tf.float32)
#ecg_labels_std= tf.convert_to_tensor(ecg_labels_std,tf.float32)
print(ecg_labels_std)
'''
datas_placeholder = tf.placeholder(tf.float32, [None, 9000])
labels_placeholder = tf.placeholder(tf.int32, [None])


# 存放DropOut参数的容器，训练时为0.25，测试时为0
dropout_placeholdr = tf.placeholder(tf.float32)


# 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
conv0 = tf.layers.conv2d(datas_placeholder, 20, 3, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool1)

# 全连接层，转换为长度为100的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, 2)

predicted_labels = tf.arg_max(logits, 1)



losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, 2),
    logits=logits
)
mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)
saver = tf.train.Saver()
train = 'True'
with tf.Session() as sess:
    if train:
        print("训练模式")
    # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())
    # 定义输入和Label以填充容器，训练时dropout为0.25
        train_feed_dict = {
            datas_placeholder: ecg_leads_std,
            labels_placeholder: ecg_leads_std,
            dropout_placeholdr: 0.25
        }
        for step in range(150):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))

for i in range(len(ecg_labels)):(
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

#and i think we must do some smoothing towards this template


model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=9000))
model.add(Activation(act_func))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, input_shape=9000))
model.add(Activation(act_func))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer = 'adam')

model.fit(x,y,batch_size=32)

'''