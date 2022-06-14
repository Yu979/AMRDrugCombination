import pandas as pd
# import deepchem as dc
import numpy as np

df = pd.read_csv('./data/drug_SMILES.csv')
smiles1 = np.array(df['SMILES']).tolist()
# create feature extractor
# featurizer = dc.feat.CircularFingerprint()
# feature = featurizer.featurize(smiles)
print(1)