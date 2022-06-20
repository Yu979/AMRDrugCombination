import pandas as pd
import numpy as np
import re
from keras.utils import to_categorical

cell_path = './data/DATABASE/comb_data.csv'
smiles_path = './data/CFfeature.npy'
drug_path = './data/drug_feature.csv'
pretrain_drug_path = './data/DATABASE/comb_data.csv'

def build_cell_line_dict(cell_path):
    drug_data = pd.read_csv(cell_path)
    a = drug_data['cell_line'].unique()
    b = ['E. coli BW25113', 'E. coli iAi1', 'ST LT2', 'ST14028', 'PAO1', 'PA14']
    b_number = [1995, 1996, 1997, 1998, 1999, 2000]
    fine_tune_vec = dict(zip(b, b_number))
    char_to_int = dict((c, i) for i, c in enumerate(a))
    int_to_char = dict((i, c) for i, c in enumerate(a))
    char_to_int.update(fine_tune_vec)
    return char_to_int


def build_drug_dict(smiles_path, drug_path):
    smiles = np.load(smiles_path, allow_pickle=True)
    drug = pd.read_csv(drug_path)
    drug = drug['Drug'].values
    smiles_vec = dict(zip(drug, smiles))
    smiles_vec = {key: val for key, val in smiles_vec.items() if val.shape != (0,)}
    return smiles_vec


def build_pretrain_data(cell_line_dict, drug_dict, pretrain_drug_path):
    drug1_feature = []
    drug2_feature = []
    cell_line = []
    label = []
    pretrain_drug_df = pd.read_csv(pretrain_drug_path)

    for s, r, m, n in zip(pretrain_drug_df['drug1'], pretrain_drug_df['drug2'], pretrain_drug_df['cell_line_number'], pretrain_drug_df['label']):
        if (pd.isna(r)) and (s in drug_dict):
            drug2_feature.append(np.zeros(1024))
            drug1_feature.append(drug_dict[s])
            cell_line.append(m)
            label.append(n)
        elif (s in drug_dict) and (r in drug_dict):
            drug1_feature.append(drug_dict[s])
            drug2_feature.append(drug_dict[s])
            cell_line.append(m)
            label.append(n)
    # for s in pretrain_drug_df['drug2']:
    #     if s in drug_dict:
    #         if pd.isna(s):
    #             drug2_feature.append(np.zeros(1024))
    #         else:
    #             drug2_feature.append([drug_dict[s]])
    return drug1_feature, drug2_feature, cell_line, label


def build_pretrain_Xy(drug1_feature, drug2_feature, cell_line, label):
    drug1_feature = np.array(drug1_feature)
    drug2_feature = np.array(drug2_feature)
    cell_line = np.array(cell_line)
    cell_line = to_categorical(cell_line, 2001)
    X = np.hstack((drug1_feature, drug2_feature, cell_line))
    y = np.array(label)
    return X, y


if __name__ == '__main__':
    cell_line_dict = build_cell_line_dict(cell_path)
    drug_dict = build_drug_dict(smiles_path, drug_path)
    drug1_feature, drug2_feature, cell_line, label = build_pretrain_data(cell_line_dict, drug_dict, pretrain_drug_path)
    X, y = build_pretrain_Xy(drug1_feature, drug2_feature, cell_line, label)
    print('end')