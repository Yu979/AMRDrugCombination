import pandas as pd
import numpy as np
import re
from keras.utils import to_categorical
import json

cell_path = './data/DATABASE/comb_data.csv'
smiles_path = './data/CFfeature.npy'
drug_path = './data/drug_feature.csv'
pretrain_drug_path = './data/DATABASE/comb_data.csv'
cell_name_path = './data/drugcomb_cell_lines.txt'
CCLE_expression_path = './data/DATABASE/CCLE_gene_cn.csv'


def find_cell_expression(name_path, expression_path):
    with open(name_path, 'r') as cell_name:
        df = pd.read_json(cell_name)
        cell_name.close()


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
    pretrain_drug_df = pretrain_drug_df.drop_duplicates(subset=['drug1', 'drug2', 'cell_line'], keep="last")
    # set samples number
    pretrain_drug_df = pretrain_drug_df.sample(n=100000, random_state=50)
    for s, r, m, n in zip(pretrain_drug_df['drug1'], pretrain_drug_df['drug2'], pretrain_drug_df['cell_line_number'], pretrain_drug_df['label']):
        if (pd.isna(r)) and (s in drug_dict):
            drug2_feature.append(np.zeros(1024))
            drug1_feature.append(drug_dict[s])
            cell_line.append(m)
            label.append(n)
        elif (s in drug_dict) and (r in drug_dict):
            drug1_feature.append(drug_dict[s])
            drug2_feature.append(drug_dict[r])
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
    y = np.array(to_categorical(label, 3))
    return X, y


if __name__ == '__main__':
    # compound_iso_smiles = []
    # df = pd.read_csv('./data/drug_SMILES.csv')
    # smiles = df['SMILES'].values
    # drug = df['Drug'].values
    # smiles_vec = dict(zip(drug, smiles))
    # compound_iso_smiles = list(smiles_vec.values())
    # compound_iso_smiles = set(compound_iso_smiles)
    # compound_iso_smiles.pop()
    cell_line_dict = build_cell_line_dict(cell_path)
    drug_dict = build_drug_dict(smiles_path, drug_path)
    drug1_feature, drug2_feature, cell_line, label = build_pretrain_data(cell_line_dict, drug_dict, pretrain_drug_path)
    X, y = build_pretrain_Xy(drug1_feature, drug2_feature, cell_line, label)
    print('end')
