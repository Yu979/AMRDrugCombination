import os, sys

import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt
import keras as K
import tensorflow as tf
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #specify GPU


hyperparameter_file = 'hyperparameters'  # textfile which contains the hyperparameters of the model
data_file = 'data_test_fold0_tanh.p.gz'  # pickle file which contains the data (produced with normalize.ipynb)


# Define smoothing functions for early stopping parameter
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


exec(open(hyperparameter_file).read())
# Load data
# tr = 60% of data for training during hyperparameter selection
# val = 20% of data for validation during hyperparameter selection
#
# train = tr + val = 80% of data for training during final testing
# test = remaining left out 20% of data for unbiased testing
#
# splitting and normalization was done with normalize.ipynb

file = gzip.open(data_file, 'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()


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
# run model for hyperparameter selection
hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))
val_loss = hist.history['val_loss']
model.reset_states()
