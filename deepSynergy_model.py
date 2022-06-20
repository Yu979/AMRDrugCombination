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
from keras.layers import Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" #specify GPU
TF_ENABLE_ONEDNN_OPTS=0
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


def build_model(hyperparameter_path):
    exec(open(hyperparameter_path).read())
    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True))
    set_session(tf.Session(config=config))
    model = Sequential()
    for i in range(len(layers)):
        if i==0:
            model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func,
                            kernel_initializer='he_normal'))
            model.add(Dropout(float(input_dropout)))
        elif i==len(layers)-1:
            model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
        else:
            model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
            model.add(Dropout(float(dropout)))
    model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))
    return model


def train(model, hyperparameter_path, X_train, y_train, X_val, y_val):
    exec(open(hyperparameter_path).read())
    # run model for hyperparameter selection
    hist1 = model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))
    return hist1


def test(model, X_train, y_train, X_test, y_test):
    average_over = 15
    mov_av = moving_average(np.array(val_loss), average_over)
    smooth_val_loss = np.pad(mov_av, int(average_over / 2), mode='edge')
    epo = np.argmin(smooth_val_loss)
    hist2 = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test))
    return hist2


if __name__ == '__main__':
    cell_line_dict = build_cell_line_dict(cell_path)
    drug_dict = build_drug_dict(smiles_path, drug_path)
    drug1_feature, drug2_feature, cell_line, label = build_pretrain_data(cell_line_dict, drug_dict, pretrain_drug_path)
    del drug_dict, cell_line_dict
    X, y = build_pretrain_Xy(drug1_feature, drug2_feature, cell_line, label)
    del drug1_feature, drug2_feature, cell_line, label
    X_train, X_test, X_val, y_train, y_test, y_val = build_dataset(X, y)
    del X, y
    model = build_model(hyperparameter_path)
    hist_train = train(model, hyperparameter_path, X_train, y_train, X_val, y_val)
    val_loss = hist_train.history['val_loss']
    model.reset_states()
    hist_test = test(model, X_train, y_train, X_test, y_test)
    test_loss = hist_test.history['val_loss']
