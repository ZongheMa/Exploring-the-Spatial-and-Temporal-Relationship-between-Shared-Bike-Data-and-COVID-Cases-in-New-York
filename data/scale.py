import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
print(os.getcwd())

df = pd.read_csv('export data for spatial analysis/df_bst_GWR.csv')

scale = MinMaxScaler()
df['covid_cases'] = scale.fit_transform(df['covid_cases'].values.reshape(-1, 1))
df['trip count'] = scale.fit_transform(df['trip count'].values.reshape(-1, 1))

print(df['covid_cases'].describe())
print(df['trip count'].describe())

df.to_csv('export data for spatial analysis/df_bst_GWR.csv')
