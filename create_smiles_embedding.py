import pandas as pd
import deepchem as dc
import numpy as np

df = pd.read_csv('./data/drug_SMILES.csv')
smiles = np.array(df['SMILES']).tolist()
# create feature extractor
featurizer = dc.feat.CircularFingerprint(size=1024, radius=4)
print('start creating features...')
feature = featurizer.featurize(smiles)
print(type(feature))
df2 = pd.DataFrame(feature, columns=['feature'])
df = pd.concat([df, df2], axis=1)
df.to_csv('./data/drug_feature.csv')
print(feature)
print(feature[0].shape)
print((np.array(feature)).shape)
print('end')
