import pandas as pd
from data_clean import merge_csv






df = merge_csv('data/Covid_cases/nyc git by zipcode')
df.drop_duplicates()
df['End date'] = pd.to_datetime(df['End date'], format='%m/%d/%Y')
df['End date'] = df['End date'].dt.strftime('%Y-%m-%d')
df.rename(columns={'End date': 'date'}, inplace=True)
df.to_csv('data/Covid_cases/covid_nyc_byzipcode.csv', encoding='utf-8')
df = df.dropna(axis=0, how='all', inplace=True)


print(df.head())
print(df.tail())
print(df.shape)
print(df.describe())