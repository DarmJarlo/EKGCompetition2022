from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models.resnet import resnet_50,resnet_mini,resnet_50_1d
import config
import math
import sys
sys.path.append("..")
from utils import wavelet, compare_plot, normalize,median_filter, butterworth,relength
sys.path.append("..")
import numpy as np
from wettbewerb import load_references

"this file is about the resnet training"
#ResNet part code adopted from https://github.com/calmisential/TensorFlow2.0_ResNet
def load_model():
    #choose 1d or 2d model
    if config.Oned != True:
        model = resnet_50()
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    else:
        model = resnet_50_1d()
        model.build(input_shape=(None, 9000, 1))
    model.summary()
    if config.FurtherTrain ==True:
        model = tf.saved_model.load('Keras_models/new_model')
     # print the network structure
    return model


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)



    'load data and make them the same length'
    ecg_leads, ecg_labels, fs, ecg_names = load_references()
    ecg_leads_std,Label_set, extra_index = relength(ecg_leads, ecg_labels)

    'Denoising : wavelet and butterworth low '
    #ecg_leads = wavelet(ecg_leads)
    #ecg_leads = butterworth(ecg_leads_de)

    'trainset and validationset split'
    mats = len(ecg_leads_std)
    X = ecg_leads_std
    X= np.float32(X)
    Label_set = np.float64(Label_set)
    train_len = 0.8 # Choice of training size
    var = int(train_len * (mats))
    X_train = X[:var]
    Y_train = Label_set[:var]
    print('tttttttttttttttt',Y_train)
    X_val = X[var:]
    Y_val = Label_set[var:]
    'reshape 1d or 2d '
    if config.Oned != True:
        X_train = np.reshape(X_train, (X_train.shape[0], config.image_height, config.image_width, config.channels))
        X_val = np.reshape(X_val, (X_val.shape[0], config.image_height, config.image_width, config.channels))
        image_size = (config.image_height, config.image_width, config.channels)
    else:
        X_train = np.reshape(X_train, (X_train.shape[0], 9000,1))
        X_val = np.reshape(X_val, (X_val.shape[0], 9000,1))
        image_size = (9000,1)


    'transform to tensor'
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    Y_train = tf.data.Dataset.from_tensor_slices(Y_train)
    X_val = tf.data.Dataset.from_tensor_slices(X_val)
    Y_val = tf.data.Dataset.from_tensor_slices(Y_val)
    train_dataset = tf.data.Dataset.zip((X_train, Y_train))
    valid_dataset = tf.data.Dataset.zip((X_val, Y_val))

    train_count = len(train_dataset)
    print("traincount",np.array(train_count).shape)
    'batch shuffle'
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    # create model
    model = load_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    alpha = 0.01
    accu = 0
    optimizer_1 = tf.keras.optimizers.Adadelta(learning_rate=alpha)

    '''optimizer_1 = tf.keras.optimizers.Adagrad(
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )'''
    #optimizer_1= tf.keras.optimizers.Adam(learning_rate=0.2)
    train_loss = tf.keras.metrics.Mean(name='train_loss')#this mean only calculate the  mean of the loss
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels,epoch):
        with tf.GradientTape() as tape:
            feature1,feature2,feature3,feature4,predictions,feature4_pooled = model(images, training=True)#dont forget here we are inputing a whole batch
            print('oooooooooooo',epoch,feature4_pooled)
            #print(predictions)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)


        optimizer_1.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        return gradients

    @tf.function
    def valid_step(images, labels):
        feature1,feature2,feature3,feature4,predictions,feature4_pooled = model(images, training=False)
        #predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)



    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        'adaptive learning rate for adadelta'
        
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
        tf.saved_model.save(model, 'Keras_models/new_model')
    '''checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                   monitor='val_accuracy',
                                   save_weights_only=False, period=1, verbose=1, save_best_only=False)'''
    tf.saved_model.save(model, 'Keras_models/new_model')
    #model.save_weights(filepath=config.save_model_dir, save_format='tf')
    #tf.saved_model.save(model, config.save_model_dir)
