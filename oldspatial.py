import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysal as ps
from data_clean import merge_csv
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.wkt import loads

# df = merge_csv('data/SharedBike/nyc')
# agg_func = {'start station name': 'first', 'start station latitude': 'first', 'start station longitude': 'first',
#             'trip count': 'sum', 'tripduration_sum(mins)': 'sum', 'geometry': 'first', 'CASE_COUNT': 'sum',
#             'HOSPITALIZED_COUNT': 'sum',
#             'DEATH_COUNT': 'sum', 'BX_CASE_COUNT': 'sum', 'BX_DEATH_COUNT': 'sum', 'BK_CASE_COUNT': 'sum',
#             'BK_DEATH_COUNT': 'sum', 'MN_CASE_COUNT': 'sum',
#             'MN_DEATH_COUNT': 'sum', 'QN_CASE_COUNT': 'sum', 'QN_DEATH_COUNT': 'sum', 'SI_CASE_COUNT': 'sum',
#             'SI_DEATH_COUNT': 'sum'}
#
# df = df.groupby(['start station id', 'date']).agg(agg_func).reset_index()
# df.drop(['geometry'], axis=1, inplace=True)
# df['date'] = pd.to_datetime(df['date'])
# df['geometry'] = df.apply(lambda x: Point(x['start station longitude'], x['start station latitude']), axis=1)
# gdf = gpd.GeoDataFrame(df, geometry='geometry', crs={'init': 'epsg:4326'})
#
# nyc = gpd.read_file('data/geo/Geography-resources/MODZCTA_2010.shp')
# nyc['MODZCTA'] = nyc['MODZCTA'].astype('int64')
#
# #
# # df = merge_csv('data/SharedBike/nyc', usecols=['date', 'start station id', 'start station name',
# #                                                'start station latitude', 'start station longitude', 'trip count',
# #                                                'tripduration_sum(mins)', 'tripduration_mean(mins)',
# #                                                'usertype_member_count', 'usertype_casual_count'])
# # df['date'] = pd.to_datetime(df['date'])
# # df['date'] = df['date'].dt.strftime('%Y-%m-%d')
# #
# # df = df[df['date'] == '2020-05-01']
#
# dfzip = pd.read_csv('data/Covid_cases/covid_nyc_byzipcode_T.csv', index_col=0, dtype={'MODZCTA': 'int64'})
# merged = pd.merge(nyc, dfzip, on='MODZCTA', how='left')
# merged = merged.rename(columns=lambda x: x.split(".")[0])
# merged = merged.loc[:, ~merged.columns.duplicated()]
#
# print(merged.columns)
# print(merged.shape)

# merged.plot(column='2020-05-01', cmap='OrRd', legend=True, figsize=(10, 10))
#
# plt.axis('off')
# plt.show()

# df['geometry'] = df.apply(lambda x: Point(x['start station longitude'], x['start station latitude']), axis=1)
# gdf = gpd.GeoDataFrame(df, geometry='geometry', crs={'init': 'epsg:4326'})




df = pd.read_csv('data/Covid_cases/covid_nyc_byzipcode.csv')



# df_zip = df_zip.fillna(0, inplace=True)
# # df_zip['date'] = pd.to_datetime(df_zip['date']).dt.strftime('%Y-%m-%d')
#
# stations = df_zip.columns[7:]
# df_melt = pd.melt(df_zip, id_vars=['date'], value_vars=stations, var_name='station', value_name='COVID_CASE_COUNT')
#
# df_pivot = pd.pivot(df_melt, index='variable', columns='date', values='COVID_CASE_COUNT').reset_index().rename_axis(
#     None, axis=1)
#
# # 对 df2 的列名进行修改
# df_pivot.columns = ['station'] + list(map(str, df_pivot.columns[1:]))
