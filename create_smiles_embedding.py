import pandas as pd
import deepchem as dc
import numpy as np

df = pd.read_csv('./data/drug_SMILES.csv')
smiles = np.array(df['SMILES']).tolist()
# create feature extractor
featurizer = dc.feat.CircularFingerprint()
print('start creating features...')
feature = featurizer.featurize(smiles)
df['feature'] = feature
df.to_csv('./data/drug_feature.csv')
print('end')
