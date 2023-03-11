import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysal as ps
from data_clean import merge_csv
from shapely.geometry import Point

df = merge_csv('data/SharedBike/nyc')
agg_func = {'start station id': 'count', 'trip count': 'sum', 'tripduration_sum(sec)': 'sum',
            'tripduration_mean(sec)': 'sum', 'tripduration_sum(mins)': 'sum', 'tripduration_mean(mins)': 'sum',
            'CASE_COUNT': 'sum', 'HOSPITALIZED_COUNT': 'sum', 'DEATH_COUNT': 'sum',
            'BX_CASE_COUNT': 'sum', 'BX_DEATH_COUNT': 'sum', 'BK_CASE_COUNT': 'sum', 'BK_DEATH_COUNT': 'sum',
            'MN_CASE_COUNT': 'sum', 'MN_DEATH_COUNT': 'sum', 'QN_CASE_COUNT': 'sum', 'QN_DEATH_COUNT': 'sum',
            'SI_CASE_COUNT': 'sum', 'SI_DEATH_COUNT': 'sum'}
df_temporal = df.groupby('date').agg(agg_func)
df_temporal.index = pd.to_datetime(df_temporal.index)
print('\n{:=^60s}'.format('df_temporal'))
print(df_temporal.shape)
print(df_temporal.head())
print(df_temporal.iloc[0])
print(df_temporal.describe())
print(df_temporal.columns)
