import pandas as pd
import numpy as np

feature_df = pd.read_csv('./data/drug_feature.csv')
feature = feature_df['feature'].values
print(feature_df['feature'][0].shape)
print(feature.shape)
print('end')