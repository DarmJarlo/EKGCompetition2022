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
from wettbewerb import load_references
from tensorflow.keras.applications import ResNet50
from Denoise import wavelet, compare_plot, normalize,median_filter, butterworth
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


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)



    # get the original_dataset

    ecg_leads, ecg_labels, fs, ecg_names = load_references()
    #ecg_leads = wavelet(ecg_leads)
    #ecg_leads = butterworth(ecg_leads_de)
    #make them the same length
    ecg_labels_std = []
    ecg_leads_std = []
    ecg_leads_extra = []
    ecg_labels_extra = []
    for index in range(len(ecg_labels)):
        if ecg_labels[index] == 'N':
            ecg_labels_std.append(0)
        elif ecg_labels[index] == 'A':
            ecg_labels_std.append(1)
        elif ecg_labels[index] == 'O':
            ecg_labels_std.append(2)
        elif ecg_labels[index] == '~':
            ecg_labels_std.append(3)

        if len(ecg_leads[index]) < 9000:
            lowiter= 9000//len(ecg_leads[index])
            print(lowiter)
            for i in range(lowiter):
                print(ecg_leads[index].shape)
                ecg_leads[index]=np.append(ecg_leads[index],ecg_leads[index])
                print('dadadad',ecg_leads[index].shape)
            ecg_leads[index]=ecg_leads[index][0:9000]
            print(len(ecg_leads[index]))
        elif len(ecg_leads[index]>9000):
            if len(ecg_leads[index]<=18000):
                ecg_leads[index]=ecg_leads[index][0:9000]
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                ecg_labels_extra.append(ecg_labels_std[index])
            elif len(ecg_leads[index]>18000):
                iter = len(ecg_leads[index])//9000
                ecg_leads[index]=ecg_leads[index][:9000]
                for i in range(1,iter):
                    start = 9000*i
                    end = 9000*(i+1)
                    ecg_leads_extra.append(ecg_leads[start:end])
                    ecg_labels_extra.append(ecg_labels_std[index])
                ecg_leads_extra.append(ecg_leads[index][-9000:])
                ecg_labels_extra.append(ecg_labels_std[index])

    ecg_labels_std = ecg_labels_std + ecg_labels_extra
    ecg_leads_std = ecg_leads + ecg_leads_extra

    print(ecg_labels_extra)
    #form the label as one-hot

    Label_set = np.zeros((len(ecg_labels_std), 4))

    for i in range(len(ecg_labels_std)):
        print('111111',i,ecg_labels_std[i])
        dummy = np.zeros(4)
        dummy[int(ecg_labels_std[i])] = 1
        Label_set[i, :] = dummy



    #compare_plot(ecg_leads[10],ecg_leads_de[10])
    mats = len(ecg_leads_std)
    # print(len(mats))


    #data transfer
    X = ecg_leads_std
    X= np.float32(X)
    Label_set = np.float64(Label_set)
    train_len = 0.8 # Choice of training size
    var = int(train_len * (mats))
    X_train = X[:var]
    Y_train = Label_set[:var]

    # Y_train = Label_set[:int(train_len * (mats))]
    print(Y_train)
    # X_val = X[int(train_len * (mats)):]
    # Y_val = Label_set[int(train_len * (mats)):]
    X_val = X[var:]
    Y_val = Label_set[var:]
    # reshape input to be [samples, tensor shape (30 x 300)]
    n = 50
    m = 180
    c = 1  # number of channels

    X_train = np.reshape(X_train, (X_train.shape[0], n, m, c))
    X_val = np.reshape(X_val, (X_val.shape[0], n, m, c))
    image_size = (n, m, c)

    # create model
    model = get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    #optimizer = tf.keras.optimizers.Adadelta(learning_rate=1) #10 iteration 0.81  every iteration has better result. BUt maybe overfitting
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=1,
        initial_accumulator_value=0.1,
        epsilon=1e-07,
        name="Adagrad"
    )
    #optimizer= tf.keras.optimizers.Adam(learning_rate=0.1)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            feature1,feature2,feature3,feature4,predictions = model(images, training=True)
            print(predictions)
            print(labels)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        print("gradients",gradients)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        feature1,feature2,feature3,feature4,predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)


    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    Y_train = tf.data.Dataset.from_tensor_slices(Y_train)
    X_val = tf.data.Dataset.from_tensor_slices(X_val)
    Y_val = tf.data.Dataset.from_tensor_slices(Y_val)
    train_dataset = tf.data.Dataset.zip((X_train, Y_train))
    valid_dataset = tf.data.Dataset.zip((X_val, Y_val))

    train_count = len(train_dataset)
    print(train_count)

    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    print('ddddddddddddddddd',train_dataset)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     config.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / config.BATCH_SIZE),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  config.EPOCHS,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))
    checkpointer = ModelCheckpoint(filepath="Keras_models/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                   monitor='val_accuracy',
                                   save_weights_only=False, period=1, verbose=1, save_best_only=False)
    model.save('Keras_models/new_model')
    #model.save_weights(filepath=config.save_model_dir, save_format='tf')
    #tf.saved_model.save(model, config.save_model_dir)
