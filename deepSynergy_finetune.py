import os, sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from pretrain_data_process import *
from deepSynergy_model import *
import matplotlib.pyplot as plt
import keras as K
import tensorflow as tf
from keras import backend
from tensorflow.python.keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

cell_path = './data/DATABASE/comb_data.csv'
smiles_path = './data/CFfeature.npy'
drug_path = './data/drug_feature.csv'
finetune_data_path = './data/train_data.xlsx'
nature_apply_path = './data/2nature_apply.xlsx'


def build_finetune_combdata(combpath):
    df = pd.read_excel(combpath, sheet_name="Supp Table 2")
    df = df.iloc[:, [1, 2, 8, 9, 10, 11, 12, 13]]
    df.columns = df.iloc[2, :].values
    df = df.iloc[3:, :]
    df = df.drop(df.tail(3).index)
    cell_line_dict = build_cell_line_dict(cell_path)
    drug_dict = build_drug_dict(smiles_path, drug_path)
    df2 = pd.DataFrame(columns=['drug1', 'drug2', 'cell_line', 'label'])
    for index, row in df.iterrows():
        # print(row['E. coli BW25113'])
        if row['E. coli BW25113'] == 'Synergy':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['E. coli BW25113'], 2]
        elif row['E. coli BW25113'] == 'Antagonism':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['E. coli BW25113'], 0]
        if row['E. coli iAi1'] == 'Synergy':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['E. coli iAi1'], 2]
        elif row['E. coli iAi1'] == 'Antagonism':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['E. coli iAi1'], 0]
        if row['ST LT2'] == 'Synergy':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['ST LT2'], 2]
        elif row['ST LT2'] == 'Antagonism':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['ST LT2'], 0]
        if row['ST14028'] == 'Synergy':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['ST14028'], 2]
        elif row['ST14028'] == 'Antagonism':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['ST14028'], 0]
        if row['PAO1'] == 'Synergy':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['PAO1'], 2]
        elif row['PAO1'] == 'Antagonism':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['PAO1'], 0]
        if row['PA14'] == 'Synergy':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['PA14'], 2]
        elif row['PA14'] == 'Antagonism':
            df2.loc[len(df2.index)] = [row['Drug 1'], row['Drug 2'], cell_line_dict['PA14'], 0]
    return df2


def build_finetune_singledata(single_path):
    df = pd.read_excel(single_path, sheet_name="Supp Table 1")
    df = df.iloc[:, [0, 27, 28, 29, 30, 31, 32]]
    cell_line_dict = build_cell_line_dict(cell_path)
    df.columns = df.iloc[4, :].values
    df = df.iloc[5:, :]
    df2 = pd.DataFrame(columns=['drug1', 'drug2', 'cell_line', 'label'])
    for index, row in df.iterrows():
        if row['E. coli BW25113'] == 'S':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['E. coli BW25113'], 2]
        elif row['E. coli BW25113'] == 'R':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['E. coli BW25113'], 0]
        if row['E. coli iAi1'] == 'S':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['E. coli iAi1'], 2]
        elif row['E. coli iAi1'] == 'R':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['E. coli iAi1'], 0]
        if row['ST LT2'] == 'S':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['ST LT2'], 2]
        elif row['ST LT2'] == 'R':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['ST LT2'], 0]
        if row['ST14028'] == 'S':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['ST14028'], 2]
        elif row['ST14028'] == 'R':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['ST14028'], 0]
        if row['PAO1'] == 'S':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['PAO1'], 2]
        elif row['PAO1'] == 'R':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['PAO1'], 0]
        if row['PA14'] == 'S':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['PA14'], 2]
        elif row['PA14'] == 'R':
            df2.loc[len(df2.index)] = [row['Drug'], '', cell_line_dict['PA14'], 0]
    return df2


def build_finetune_xy(comb_data, single_data, drug_dic):
    drug1_features = []
    drug2_features = []
    cell_lines = []
    labels = []
    for s, r, m, n in zip(comb_data['drug1'], comb_data['drug2'], comb_data['cell_line'],
                          comb_data['label']):
        if (s in drug_dic) and (r in drug_dic):
            drug1_features.append(drug_dict[s])
            drug2_features.append(drug_dict[r])
            cell_lines.append(m)
            labels.append(n)
    for s, r, m, n in zip(single_data['drug1'], single_data['drug2'], single_data['cell_line'],
                          single_data['label']):
        if s in drug_dic:
            drug1_features.append(drug_dict[s])
            drug2_features.append(np.zeros(1024))
            cell_lines.append(m)
            labels.append(n)
    cell_lines = to_categorical(cell_lines, 2001)
    labels = to_categorical(labels, 3)
    finetune_X = np.hstack((drug1_features, drug2_features, cell_lines))
    finetune_Y = np.array(labels)
    return finetune_X, finetune_Y


def split_finetune_data(finetune_x, finetune_y):
    X_tr, X_test, y_tr, y_test = train_test_split(finetune_x, finetune_y, test_size=0.2, random_state=1)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=1)
    return X_tr, X_test, X_val, y_tr, y_test, y_val


def split_train_notest(finetune_x, finetune_y):
    X_tr, y_tr = finetune_x[: 1650], finetune_y[: 1650]
    X_test, y_test = finetune_x[1650: 2100], finetune_y[1650: 2100]
    X_val, y_val = finetune_x[2100:], finetune_y[2100:]
    return X_tr, X_test, X_val, y_tr, y_test, y_val


def split_strain_train_notest(finetune_x, finetune_y):
    X_tr, y_tr = finetune_x[20: 2123], finetune_y[20: 2123]
    X_test, y_test = finetune_x[2123:], finetune_y[2123:]
    X_val, y_val = finetune_x[:20], finetune_y[:20]
    return X_tr, X_test, X_val, y_tr, y_test, y_val


def build_finetune_xy_split_strain(comb_data, single_data, drug_dic):
    comb_data = pd.concat([single_data, comb_data])
    comb_data = comb_data.sort_values("cell_line", ascending=True)
    comb_data = comb_data.reset_index(drop=True)
    drug1_features = []
    drug2_features = []
    cell_lines = []
    labels = []

    for s, r, m, n in zip(comb_data['drug1'], comb_data['drug2'], comb_data['cell_line'],
                          comb_data['label']):
        if (s in drug_dic) and (r in drug_dic):
            drug1_features.append(drug_dict[s])
            drug2_features.append(drug_dict[r])
            cell_lines.append(m)
            labels.append(n)
        elif (s in drug_dic) and (r is np.nan):
            drug1_features.append(drug_dict[s])
            drug2_features.append(np.zeros(1024))
            cell_lines.append(m)
            labels.append(n)

    cell_lines = to_categorical(cell_lines, 2001)
    labels = to_categorical(labels, 3)
    finetune_X = np.hstack((drug1_features, drug2_features, cell_lines))
    finetune_Y = np.array(labels)
    return finetune_X, finetune_Y


def build_finetune_xy_split_combination(comb_data, single_data, drug_dic):
    comb_data = pd.concat([single_data, comb_data])
    comb_data = comb_data.sort_values(by=["drug2"], ascending=True)
    comb_data = comb_data.reset_index(drop=True)
    drug1_features = []
    drug2_features = []
    cell_lines = []
    labels = []

    for s, r, m, n in zip(comb_data['drug1'], comb_data['drug2'], comb_data['cell_line'],
                          comb_data['label']):
        if (s in drug_dic) and (r in drug_dic):
            drug1_features.append(drug_dict[s])
            drug2_features.append(drug_dict[r])
            cell_lines.append(m)
            labels.append(n)
        elif (s in drug_dic) and (r is np.nan):
            drug1_features.append(drug_dict[s])
            drug2_features.append(np.zeros(1024))
            cell_lines.append(m)
            labels.append(n)

    cell_lines = to_categorical(cell_lines, 2001)
    labels = to_categorical(labels, 3)
    finetune_X = np.hstack((drug1_features, drug2_features, cell_lines))
    finetune_Y = np.array(labels)
    return finetune_X, finetune_Y


def train(model, X_train, y_train, X_val, y_val):
    epochs = 1000
    batch_size = 64
    act_func = tf.nn.relu
    dropout = 0.5
    input_dropout = 0.2
    eta = 0.0001
    norm = 'tanh'
    # run model for hyperparameter selection
    steps_per_epoch = len(X_train) // (batch_size)
    hist1 = model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_size,
                      validation_data=(X_val, y_val), steps_per_epoch=steps_per_epoch)
    return hist1


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def test(model, X_train, y_train, X_test, y_test):
    average_over = 15
    batch_size = 64
    steps_per_epoch = len(X_train) // (batch_size)
    mov_av = moving_average(np.array(val_loss), average_over)
    smooth_val_loss = np.pad(mov_av, int(average_over / 2), mode='edge')
    epo = np.argmin(smooth_val_loss)
    hist2 = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test),
                      steps_per_epoch=steps_per_epoch)
    y_pre = model.predict(X_test)
    y_in_class = np.rint(y_pre)
    cr = classification_report(y_test, y_in_class)
    print(cr)
    return hist2


def build_nature_apply_data(nature_apply_path, drug_dic):
    df = pd.read_excel(nature_apply_path)
    drug1_features = []
    drug2_features = []
    cell_lines = []
    labels = []
    for s, r, m, n in zip(df['drug1'], df['drug2'], df['cell_line'],
                          df['label']):
        if (s in drug_dic) and (r in drug_dic):
            drug1_features.append(drug_dict[s])
            drug2_features.append(drug_dict[r])
            cell_lines.append(m)
            labels.append(n)
    cell_lines = to_categorical(cell_lines, 2001)
    labels = to_categorical(labels, 3)
    nature_apply_X = np.hstack((drug1_features, drug2_features, cell_lines))
    nature_apply_Y = np.array(labels)
    return nature_apply_X, nature_apply_Y


if __name__ == '__main__':
    drug_dict = build_drug_dict(smiles_path, drug_path)
    combdata = build_finetune_combdata(finetune_data_path)
    singledata = build_finetune_singledata(finetune_data_path)
    # X, y = build_finetune_xy(combdata, singledata, drug_dict)
    nature_x, nature_y = build_nature_apply_data(nature_apply_path, drug_dict)
    X, y = build_finetune_xy_split_strain(combdata, singledata, drug_dict)
    # X_train, X_test, X_val, y_train, y_test, y_val = split_finetune_data(X, y)
    # X_train, X_test, X_val, y_train, y_test, y_val = split_train_notest(X, y)
    X_train, X_test, X_val, y_train, y_test, y_val = split_strain_train_notest(X, y)
    X_train, X_test, X_val = expand_dim(X_train, X_test, X_val)
    # pretrain_model = K.models.load_model('./model/NNpretrain.h5')
    pretrain_model = K.models.load_model('./model/CNNpretrain.h5')
    for layer in pretrain_model.layers[:2]:
        layer.trainable = False
    eta = 0.001
    pretrain_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=K.metrics.categorical_accuracy)
    hist_train = train(pretrain_model, X_train, y_train, X_val, y_val)
    val_loss = hist_train.history['val_loss']
    pretrain_model.reset_states()
    hist_test = test(pretrain_model, X_train, y_train, X_test, y_test)
    test_loss = hist_test.history['val_loss']

    X_com, y_com = build_finetune_xy_split_combination(combdata, singledata, drug_dict)
    X_train_com, X_test_com, X_val_com, y_train_com, y_test_com, y_val_com = split_train_notest(X, y)
    X_train_com, X_test_com, X_val_com = expand_dim(X_train_com, X_test_com, X_val_com)
    pretrain_model = K.models.load_model('./model/CNNpretrain.h5')
    for layer in pretrain_model.layers[:2]:
        layer.trainable = False
    eta = 0.001
    pretrain_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=K.metrics.categorical_accuracy)
    hist_train2 = train(pretrain_model, X_train_com, y_train_com, X_val_com, y_val_com)
    val_loss2 = hist_train2.history['val_loss']
    pretrain_model.reset_states()
    hist_test2 = test(pretrain_model, X_train_com, y_train_com, X_test_com, y_test_com)
    test_loss2 = hist_test2.history['val_loss']

    pretrain_model = K.models.load_model('./model/CNNpretrain.h5')
    for layer in pretrain_model.layers[:2]:
        layer.trainable = False
    eta = 0.001
    pretrain_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=K.metrics.categorical_accuracy)
    steps_per_epoch = len(X_train) // 64
    model.fit(X_train, y_train, epochs=1000, shuffle=True, batch_size=64,
              steps_per_epoch=steps_per_epoch)
    y_pre3 = model.predict(nature_x)
    y_in_class3 = np.rint(y_pre3)
    cr3 = classification_report(nature_y, y_in_class3)
    print(cr3)
    print('end')
