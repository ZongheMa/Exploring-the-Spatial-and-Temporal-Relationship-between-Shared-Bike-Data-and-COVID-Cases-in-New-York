import geopandas as gpd
import numpy.linalg
import pandas as pd
import numpy as np
from data_clean import merge_csv
from tqdm import tqdm
import datetime
import mgwr
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW


bst = pd.read_csv('data/SharedBike/bst/cleaned-201901-bluebikes-tripdata.csv')
print(bst.columns)
df = merge_csv('data/SharedBike/bst', usecols=['date', 'start station id', 'start station name',
                                                  'start station latitude', 'start station longitude',
                                                  'trip count', 'tripduration_sum(mins)',
                                                  'tripduration_mean(mins)', 'usertype_member_count',
                                                  'usertype_casual_count','CASE_COUNT'])
# bikes = gpd.GeoDataFrame(bikes,
#                          geometry=gpd.points_from_xy(bikes['start station longitude'],
#                                                      bikes['start station latitude']))
# bikes = bikes.set_crs(epsg=4326, allow_override=True)
# gdf = gpd.GeoDataFrame()
# gdf['CASE_COUNT'] = np.nan
# count = 0
# for i in tqdm(bikes['date'].unique()):
#     if count == 14:
#         break
#
#     if i in covid_byzip.columns:
#         print(i)
#         sjoin = gpd.sjoin(bikes[bikes['date'] == i], covid_byzip[['MODZCTA', i, 'geometry']], how='left',
#                           op='intersects')
#         sjoin = sjoin[pd.notnull(sjoin[i])]
#         sjoin = sjoin.drop(columns=['index_right'])
#         gdf = pd.concat([gdf, sjoin])
#         gdf.loc[bikes['date'] == i, 'covid_cases'] = sjoin[i].values
#         gdf.drop(columns=i, inplace=True)
#         count += 1
#         print(count)
#     else:
#         continue


# df = gdf.drop(columns=['geometry', 'MODZCTA', 'usertype_casual_count'], axis=1)
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df.set_index('date', inplace=True)

# df = df.groupby('start station id').resample('W').agg(
#     {'start station name': 'first', 'start station latitude': 'first', 'start station longitude': 'first',
#      'covid_cases': 'sum', 'trip count': 'sum', 'tripduration_sum(mins)': 'sum', 'tripduration_mean(mins)': 'sum',
#      'usertype_member_count': 'sum'})
# df = df.reset_index()

null_cols = df.columns[df.isnull().any()].tolist()
null_rows = df.isnull().any(axis=1)

# df = df.replace([np.inf, -np.inf], np.nan)
# df = df.dropna()
# print(df.isnull().sum())
# print(df.isfinite().sum())
# df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10 ** 9
# print(df.shape)
# df.to_csv('data/df_M.csv')

# coords = df[['start station latitude', 'start station longitude']]
# # coords = list(zip(gdf['start station longitude'], gdf['start station latitude']))
#
# X = df[
#     ['covid_cases', 'tripduration_sum(mins)', 'tripduration_mean(mins)', 'usertype_member_count']].astype(
#     'float16')
# y = df[['trip count']].astype(int)
# t = df[['timestamp']].astype(int)

# # Normalize the variables
# min_max_scaler = preprocessing.MinMaxScaler()
# X[X.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(X.select_dtypes(include=[np.number]))
# y[y.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(y.select_dtypes(include=[np.number]))

# print('got it')

#
# # int_col = []
# # for date in t:
# #     # date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
# #     timestamp = date.timestamp()
# #     int_val = int(timestamp)
# #     int_col.append(int_val)
# #
# # t = np.array(int_col)
# # t = t.reshape(-1, 1)
# # print('after t parse')
# # #

# # Fit MGTWR model
# sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
# bws = sel_multi.search()
# # class sel_multi:
# #     def __init__(self, bws):
# #         self.bws = bws
# #
# #
# # bws = [90, 95, 100, 120, 95, 100, 120, 95, 100, 120]
#
# # selector = sel_multi(bws)
# print('fit')
#
# mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel='gaussian', fixed=True).fit()
#
# print(mgtwr.R2)

# fit GWR model from mgwr package
df.rename(columns={'CASE_COUNT': 'covid_cases'}, inplace=True)
df = df.groupby('start station id').agg(
    {'start station name': 'first', 'start station latitude': 'first', 'start station longitude': 'first',
     'covid_cases': 'sum', 'trip count': 'sum', 'tripduration_sum(mins)': 'sum', 'tripduration_mean(mins)': 'sum',
     'usertype_member_count': 'sum'})
df = df.reset_index()

null_cols = df.columns[df.isnull().any()].tolist()
null_rows = df.isnull().any(axis=1)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# df['geometry'] = gpd.points_from_xy(df['start station longitude'], df['start station latitude'])
# df = gpd.GeoDataFrame(df, geometry='geometry')
df.to_csv('data/df for ST analysis/df_bst_GWR.csv')
# X = df[['covid_cases', 'tripduration_sum(mins)', 'tripduration_mean(mins)', 'usertype_member_count']]
# y = df[['trip count']]
#
# sel = Sel_BW(X, y, df.geometry)
# bw = sel.search()
#
# model = MGWR(df.geometry, y, bw, fixed=True).fit()
