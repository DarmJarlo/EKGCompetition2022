import tensorflow as tf
from config import NUM_CLASSES
from models.residual_block import  make_bottleneck_layer, OneD_bottleneck_layer



class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        #self.fc2 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.softmax)
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        feature1 = self.layer1(x, training=training)
        feature2 = self.layer2(feature1, training=training)
        feature3 = self.layer3(feature2, training=training)
        feature4 = self.layer4(feature3, training=training)
        output1 = self.avgpool(feature4)
        #output1= self.fc2(output2)
        #print('PPPPPPPPPPPP',output1.shape) #shape 128x2048

        output = self.fc(output1)
        #output = output/sum(output)
        return feature1,feature2,feature3,feature4,output,output1
        #return output

class ResNetType_1d(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetType_1d, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=64,
                                            kernel_size=7,
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=3,
                                               strides=2,
                                               padding="same")

        self.layer1 = OneD_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = OneD_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = OneD_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = OneD_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        #self.fc2 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.softmax)
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        feature1 = self.layer1(x, training=training)
        feature2 = self.layer2(feature1, training=training)
        feature3 = self.layer3(feature2, training=training)
        feature4 = self.layer4(feature3, training=training)
        output1 = self.avgpool(feature4)

        output = self.fc(output1)
        #output = output/sum(output)
        return feature1,feature2,feature3,feature4,output,output1


def resnet_mini():
    return ResNetTypeII(layer_params=[0,0,0,1])

def resnet_50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])
def resnet_50_1d():
    return ResNetType_1d(layer_params=[3, 4, 6, 3])

