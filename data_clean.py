# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import zipfile
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

'''# Covid-19 data from NYC Health
original_covid_nyc = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv')
covid_nyc = pd.DataFrame(original_covid_nyc,
                         columns=['date_of_interest', 'CASE_COUNT', 'HOSPITALIZED_COUNT', 'DEATH_COUNT',
                                  'BX_CASE_COUNT', 'BX_DEATH_COUNT', 'Bk_CASE_COUNT', 'Bk_DEATH_COUNT', 'MN_CASE_COUNT',
                                  'MN_DEATH_COUNT', 'QN_CASE_COUNT', 'QN_DEATH_COUNT','SI_CASE_COUNT', 'SI_DEATH_COUNT'])
covid_nyc = covid_nyc.loc[(covid_nyc['date_of_interest'] >= '03/01/2020') & (covid_nyc['date_of_interest'] <= '12/31/2022')]

print(original_covid_nyc.head())
print(covid_nyc['CASE_COUNT'].sum())


pd.plotting.scatter_matrix(covid_nyc, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()'''


# class NYC_bikeData_Clean(self):
#     self.source_folder = s
#     self.target_folder = '/Users/zonghe/Documents/Modules/Term2/CEGE0042_STDM/STDM/data/shared bike datasets/nyc'
#


# Shared bike datasets


def unzip_files(source_folder, target_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith('.zip'):
            file_path = os.path.join(source_folder, filename)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(target_folder)


source_folder = '/Users/zonghe/Downloads/'
target_folder = '/Users/zonghe/Documents/Modules/Term2/CEGE0042_STDM/STDM/data/shared bike datasets/nyc'
# unzip_files(source_folder, target_folder)


# # Merge csv files
# all_files = glob.glob(target_folder + "/*.csv")
# list_ = []
# for file in all_files:
#     df = pd.read_csv(file, index_col=None, header=0)
#     list_.append(df)
#
# frame = pd.concat(list_, axis=0, ignore_index=True)
# frame.to_csv('/Users/zonghe/Documents/Modules/Term2/CEGE0042_STDM/STDM/data/shared bike datasets/nyc_sharedbikes_merged.csv', index=False)


ori_csv = pd.read_csv('data/shared bike datasets/nyc/201910-citibike-tripdata.csv')
ori_len = len(ori_csv)
ori_csv[(ori_csv['tripduration'] > 60) & (ori_csv['tripduration'] < 21600)]
print(f'Filter out {len(ori_csv) - ori_len} invalid data（out of time data)')

ori_csv['starttime'], ori_csv['stoptime'] = pd.to_datetime(ori_csv['starttime']), pd.to_datetime(ori_csv['stoptime'])
# print(type(a['starttime'][1]))
ori_csv['starttime'], ori_csv['stoptime'] = ori_csv['starttime'].dt.strftime('%Y-%m-%d'), ori_csv[
    'stoptime'].dt.strftime('%Y-%m-%d')
# print(ori_csv['starttime'][:])

csv = pd.DataFrame()

csv['date'] = ori_csv.groupby(['starttime']).agg({'starttime': 'first'})
csv['tripcount'] = ori_csv.groupby(['starttime']).agg({'tripduration': 'count'})
csv['tripdurtion_sum'] = ori_csv.groupby(['starttime']).agg({'tripduration': 'sum'})
csv['tripcount_mean'] = ori_csv.groupby(['starttime']).agg({'tripduration': 'mean'})
csv['active_stations_id'] = ori_csv.groupby(['starttime'])['start station id'].agg(lambda x: x.unique().tolist())
csv['active_stations_name'] = ori_csv.groupby(['starttime'])['start station name'].agg(lambda x: x.unique().tolist())

unique_stations = ori_csv[['start station id']].drop_duplicates().reset_index(drop=True).sort_values(
    by='start station id')
# print(unique_stations)
sta_info = ori_csv.loc[:,
           ['start station id', 'start station name', 'start station latitude', 'start station longitude']]
# print(sta_info.columns)
merge_station = unique_stations.merge(sta_info, on='start station id', how='left').drop_duplicates().reset_index(
    drop=True)
merge_station['active_station_count'] = ori_csv.groupby(['starttime'])['start station id'].count()
# print(merge_station.head())



df = pd.DataFrame()
active_station_count = []
for date in ori_csv['starttime'].unique():
    rows = ori_csv[ori_csv['starttime'] == date]
    count = 0
    for station in rows['start station id'].unique():
        count = rows[rows['start station id'] == station]['start station id'].count()
        active_station_count.append(count)
    df = df.append({'date': date, 'active_stations_id': rows['start station id'].unique(),
                    'active_stations_name': rows['start station name'].unique(),
                    'active_stations_count': active_station_count,
                    'lon': merge_station['start station longitude'], 'lat': merge_station['start station latitude'],
                    'tripcount': len(rows), 'tripdurtion_sum(mins)': rows['tripduration'].sum() / 60,
                    'tripdurtion_mean(mins)': rows['tripduration'].mean() / 60}, ignore_index=True)

dic_list1 = []
dic_list2 = []
points_list = []
multipoints_list = []
for i in range(len(df)):
    for j in range(len(df['active_stations_id'][i])):
        dic = {'station_id': df['active_stations_id'][i][j],
               'station_name': df['active_stations_name'][i][j],
               'point': Point(df['lon'][i][j], df['lat'][i][j])
               }
        dic_list2.append(dic)

    dic_list1.append(dic_list2)

for i in range(len(dic_list1)):
    for j in range(len(dic_list1[i])):
        points_list.append(dic_list1[i][j]['point'])
    geometry = MultiPoint(points_list)
    multipoints_list.append(geometry)

df['geometry'] = multipoints_list
s_df = gpd.GeoDataFrame(df, geometry='geometry', crs={'init': 'epsg:4326'})

t_df = s_df.explode('date').explode('active_stations_id').explode('active_stations_name').explode('active_stations_count').explode('lon').explode('lat').explode('tripcount').explode('tripdurtion_sum(mins)').explode('tripdurtion_mean(mins)')
t_df.reset_index(drop=True, inplace=True)

# s_df.drop(columns=['lon', 'lat', 'active_stations_id', 'active_stations_name', 'active_stations_count'], inplace=True)
print(s_df.head())
print(t_df.head())
