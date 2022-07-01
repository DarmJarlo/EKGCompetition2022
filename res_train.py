from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from keras.callbacks import ModelCheckpoint
import config
from prepare_data import generate_datasets
import math
import sys
sys.path.append("..")
import numpy as np
from sklearn.model_selection import train_test_split
from wettbewerb import load_references
from tensorflow.keras.applications import ResNet50
from Denoise import wavelet, compare_plot, normalize,median_filter, butterworth
from data_analysis import uniform_length
#https://github.com/calmisential/TensorFlow2.0_ResNet

def get_model():
    #model = ResNet50(input_shape=(20,45,1))
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


def train_res(X, y):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # reshape input to be [samples, tensor shape (30 x 300)]
    n = 50  #90
    m = 180  #100
    c = 1  # number of channels
    #X = wavelet(X)
    #X = butterworth(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.float32(X_train)
    X_val = np.float32(X_val)
    y_train = np.float64(y_train)
    y_val = np.float64(y_val)

    X_train = np.reshape(X_train, (len(X_train), n, m, c))
    X_val = np.reshape(X_val, (len(X_val), n, m, c))
    image_size = (n, m, c)

    # create model
    model = get_model()

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
    #optimizer = tf.keras.optimizers.Adadelta(learning_rate=1) #10 iteration 0.81  every iteration has better result. BUt maybe overfitting
    #optimizer = tf.keras.optimizers.Adagrad(
    #    learning_rate=0.001,
    #    initial_accumulator_value=0.1,
    #    epsilon=1e-07,
    #    name="Adagrad"
    #)
    #optimizer= tf.keras.optimizers.Adam(learning_rate=0.1)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels,epoch):
        #print('zzzzzzzzzzzzzz')
        with tf.GradientTape() as tape:
            predictions,feature4_pooled = model(images, training=True)#dont forget here we are inputing a whole batch
            print('oooooooooooo',epoch,feature4_pooled)
            #predictions = model(images, training=True)
            #predictions = predictions.numpy()
            #predictions = predictions/np.sum(predictions)
            print(predictions)

            #print(labels)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        #print("gradients",gradients)

        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        return gradients

    @tf.function
    def valid_step(images, labels):
        predictions,feature4_pooled = model(images, training=False)
        #predictions = model(images, training=False)
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
    print("traincount",np.array(train_count).shape)

    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    #print('ddddddddddddddddd',train_dataset)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        if epoch < 6:
            alpha = epoch*0.2+0.2
        elif epoch > 15 and accu < 0.85:
            alpha = 0.2+(epoch-16*0.2)
        else:
            alpha = alpha/5
        print("alpha",alpha)
        #optimizer_1 = tf.keras.optimizers.Adam(learning_rate=alpha)
        step = 0
        for images, labels in train_dataset:
            #print("images",images.shape,labels.shape)
            step += 1
            #gradients=train_step(images, labels)
            train_step(images, labels,epoch)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     config.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / config.BATCH_SIZE),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))
            #print(gradients)

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
    model.save('Keras_models/new_model')