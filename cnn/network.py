import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import tensorflow as tf

class Network():
    def __init__(self):
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, input_shape=(9000,1),
                                  kernel_initializer=init, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.maxpool1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpool2 = MaxPooling2D()
        self.conv_layer3 = Conv2D(128, kernel_size=3, activation=act_func, kernel_initializer=init, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.maxpool3 = MaxPooling2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(500, activation=act_func, kernel_initializer=init)
        self.dense_2 = Dense(500, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(100, activation=act_func, kernel_initializer=init)
        self.dense_4 = Dense(100, activation=act_func, kernel_initializer=init)
        self.dense_5 = Dense(20, activation=act_func, kernel_initializer=init)
        self.dense_6 = Dense(20, activation=act_func, kernel_initializer=init)

