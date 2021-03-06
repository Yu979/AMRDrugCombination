import os, sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import gzip
from pretrain_data_process import *
import matplotlib.pyplot as plt
import keras as K
import tensorflow as tf
from keras import backend
from tensorflow.python.keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import LSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" #specify GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
cell_path = './data/DATABASE/comb_data.csv'
smiles_path = './data/CFfeature.npy'
drug_path = './data/drug_feature.csv'
pretrain_drug_path = './data/DATABASE/comb_data.csv'

hyperparameter_path = './config/hyperparameters.txt'  # textfile which contains the hyperparameters of the model
data_file = 'data_test_fold0_tanh.p.gz'  # pickle file which contains the data (produced with normalize.ipynb)


# Define smoothing functions for early stopping parameter
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def build_dataset(X, y):
    # Load data
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=1)
    # tr = 60% of data for training during hyperparameter selection
    # val = 20% of data for validation during hyperparameter selection
    #
    # train = tr + val = 80% of data for training during final testing
    # test = remaining left out 20% of data for unbiased testing
    return X_tr, X_test, X_val, y_tr, y_test, y_val


def build_model(X_tr):
    layers = [8182, 4096, 3]
    epochs = 100
    act_func = tf.nn.relu
    dropout = 0.5
    input_dropout = 0.2
    eta = 0.0001
    norm = 'tanh'
    exec(open(hyperparameter_path).read())
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    set_session(tf.compat.v1.Session(config=config))
    model = Sequential()
    for i in range(len(layers)):
        if i==0:
            model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func,
                           ))
            model.add(Dropout(float(input_dropout)))
        elif i==len(layers)-1:
            model.add(Dense(layers[i], activation='softmax'))
        else:
            model.add(Dense(layers[i], activation=act_func))
            model.add(Dropout(float(dropout)))
    model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5), metrics=K.metrics.categorical_accuracy)
    return model


def build_cnn(X_tr):
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    set_session(tf.compat.v1.Session(config=config))
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=16,
                     strides=1, padding='same', activation='relu', input_shape=(X_tr.shape[1],X_tr.shape[2])))
    model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(100, use_bias=True, dropout=0.1, return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=K.metrics.categorical_accuracy)
    return model


def expand_dim(X_train, X_test, X_val):
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    X_val = np.expand_dims(X_val, axis=2)

    return X_train, X_test, X_val

def train(model, X_train, y_train, X_val, y_val):
    layers = [8182, 4096, 1]
    epochs = 100
    batch_size = 64
    act_func = tf.nn.relu
    dropout = 0.5
    input_dropout = 0.2
    eta = 0.0001
    norm = 'tanh'
    # run model for hyperparameter selection
    steps_per_epoch = len(X_train) // (batch_size)
    hist1 = model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_size, validation_data=(X_val, y_val), steps_per_epoch=steps_per_epoch)
    return hist1


def test(model, X_train, y_train, X_test, y_test):
    average_over = 15
    batch_size = 64
    steps_per_epoch = len(X_train) // (batch_size)
    mov_av = moving_average(np.array(val_loss), average_over)
    smooth_val_loss = np.pad(mov_av, int(average_over / 2), mode='edge')
    epo = np.argmin(smooth_val_loss)
    hist2 = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test), steps_per_epoch=steps_per_epoch)
    return hist2


if __name__ == '__main__':
    cell_line_dict = build_cell_line_dict(cell_path)
    drug_dict = build_drug_dict(smiles_path, drug_path)
    drug1_feature, drug2_feature, cell_line, label = build_pretrain_data(cell_line_dict, drug_dict, pretrain_drug_path)
    del drug_dict, cell_line_dict
    X, y = build_pretrain_Xy(drug1_feature, drug2_feature, cell_line, label)
    del drug1_feature, drug2_feature, cell_line, label
    # for X, y in zip(np.array_split(X, 100), np.array_split(y, 100)):
    X_train, X_test, X_val, y_train, y_test, y_val = build_dataset(X, y)
    del X, y
    # model = build_model(X_train)
    X_train, X_test, X_val = expand_dim(X_train, X_test, X_val)
    print(X_train.shape[1])
    model = build_cnn(X_train)
    hist_train = train(model, X_train, y_train, X_val, y_val)
    val_loss = hist_train.history['val_loss']
    model.reset_states()
    # model.save('./model/NNpretrain.h5')
    model.save('./model/CNNpretrain.h5')
    hist_test = test(model, X_train, y_train, X_test, y_test)
    test_loss = hist_test.history['val_loss']
