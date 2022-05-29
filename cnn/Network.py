import tensorflow.keras as keras
from keras import Model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout,Activation
import tensorflow as tf

class Network(keras.Model):
    '''https://www.youtube.com/watch?v=WvoLTXIjBYU
        'https://keras.io/api/layers/pooling_layers/max_pooling2d/'''
    def __init__(self):
        '''
        self.model.add( Conv2D(32, kernel_size=3, input_shape=(30,30,10)))
        self.model.add( Activation(act_func))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(32, kernel_size=3, input_shape=(30,30,10)))
        self.model.add(Activation(act_func))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(4, activation='softmax'))
        '''
        super().__init__()
        init = keras.initializers.GlorotNormal()
        act_func = 'relu'
        self.conv_layer1 = Conv2D(32, kernel_size=3, activation=act_func, input_shape=(20,450,1),
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
        self.dense_2 = Dense(100, activation=act_func, kernel_initializer=init)
        self.dense_3 = Dense(20, activation=act_func, kernel_initializer=init)
        self.outp = Dense(4, activation='softmax', kernel_initializer=init)
    def call(self, inputs):

        out = self.conv_layer1(input)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.conv_layer2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.conv_layer3(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        out = self.flatten(out)
        out = self.dense_1(out)
        out = self.dense_2(out)
        out = self.dense_3(out)
        predi = self.outp(out)




        return predi







