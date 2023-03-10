# coding=utf-8
import numpy as np
import glob
import pandas as pd
import os
import zipfile
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm


# # Shared bike DataSets
# url_nyc = 'https://s3.amazonaws.com/tripdata/index.html'
# url_boston = 'https://divvy-tripdata.s3.amazonaws.com/index.html'
# url_chicago = 'https://s3.amazonaws.com/hubway-data/index.html'

# # NYC Covid DataSets
# download_link('https://s3.amazonaws.com/tripdata/201306-citibike-tripdata.zip')


def get_csv_paths(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                csv_files.append(file_path)
    return csv_files


# Shared bike datasets
def unzip_files(source_folder, target_folder):
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith('.zip'):
            file_path = os.path.join(source_folder, filename)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(target_folder)


def merge_csv(source_folder, target_folder=None, file_format=None):

    all_files = glob.glob(source_folder + "/*.csv")
    list_ = []
    for file in all_files:
        df = pd.read_csv(file, index_col=None, header=0)
        list_.append(df)

    frame = pd.concat(list_, axis=0, ignore_index=True)
    if file_format == None:
        frame
    elif file_format == 'csv':
        frame.to_csv(
            target_folder + 'merged.csv',
            index=False)
    else:
        print('Please input the correct file format.')
    return frame

class BikeDataClean:
    def __init__(self, csv):
        self.ori_csv = csv

    def uniform(self):
        df = pd.read_csv(self.ori_csv,encoding='utf-8')
        if 'tripduration' not in df.columns:
            df.rename(
                columns={'start_station_id': 'start station id', 'started_at': 'starttime', 'ended_at': 'stoptime',
                         'start_station_name': 'start station name', 'start_lat': 'start station latitude',
                         'start_lng': 'start station longitude'}, inplace=True)
            df['tripduration'] = (pd.to_datetime(df['stoptime']) - pd.to_datetime(df['starttime'])).dt.seconds
            df['tripduration'] = df['tripduration'].astype(float)
            df.to_csv(self.ori_csv, index=False)

    def basic_csv(self):
        df = pd.read_csv(self.ori_csv, low_memory=False)

        # filter out invalid data
        ori_len = len(df)
        df = df[(df['tripduration'] > 60) & (df['tripduration'] < 21600)]
        print(f'Filtered out {ori_len - len(df)} invalid data (out of time range data).')

        # convert to datetime and extract date
        df['starttime'] = pd.to_datetime(df['starttime'])
        df['stoptime'] = pd.to_datetime(df['stoptime'])
        df['starttime'] = df['starttime'].dt.date.astype(str)
        df['stoptime'] = df['stoptime'].dt.date.astype(str)

        # group and aggregate
        group_cols = ['starttime', 'start station id', 'start station name',
                      'start station latitude', 'start station longitude']
        agg_dict = {'tripduration': ['count', 'sum', 'mean'], 'start station name': 'first'}
        grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped.rename(columns={'starttime_': 'starttime',
                                'start station id_': 'start station id',
                                'start station name_': 'start station name',
                                'start station latitude_': 'start station latitude',
                                'start station longitude_': 'start station longitude',
                                'tripduration_count': 'trip count',
                                'tripduration_sum': 'tripduration_sum(sec)',
                                'tripduration_mean': 'tripduration_mean(sec)'}, inplace=True)
        grouped.drop(columns=['start station name_first'], inplace=True)

        # convert units and create geometry column
        grouped['tripduration_sum(mins)'] = round(grouped['tripduration_sum(sec)'] / 60, 4)
        grouped['tripduration_mean(mins)'] = round(grouped['tripduration_mean(sec)'] / 60, 4)
        grouped['geometry'] = [Point(xy) for xy in zip(grouped['start station longitude'],
                                                       grouped['start station latitude'])]
        gdf = gpd.GeoDataFrame(grouped, geometry='geometry', crs='epsg:4326')

        return gdf

    def complex_csv(self):

        ori_csv = pd.read_csv(self.ori_csv)
        ori_len = len(ori_csv)
        ori_csv = ori_csv[(ori_csv['tripduration'] > 60) & (ori_csv['tripduration'] < 21600)]
        print(f'Filter out {len(ori_csv) - ori_len} invalid data（out of time data)')

        csv = pd.DataFrame()
        csv['date'] = ori_csv.groupby(['starttime']).agg({'starttime': 'first'})
        csv['tripcount'] = ori_csv.groupby(['starttime']).agg({'tripduration': 'count'})
        csv['tripdurtion_sum'] = ori_csv.groupby(['starttime']).agg({'tripduration': 'sum'})
        csv['tripcount_mean'] = ori_csv.groupby(['starttime']).agg({'tripduration': 'mean'})
        csv['active_stations_id'] = ori_csv.groupby(['starttime'])['start station id'].agg(
            lambda x: x.unique().tolist())
        csv['active_stations_name'] = ori_csv.groupby(['starttime'])['start station name'].agg(
            lambda x: x.unique().tolist())

        unique_stations = ori_csv[['start station id']].drop_duplicates().reset_index(drop=True).sort_values(
            by='start station id')
        # print(unique_stations)
        sta_info = ori_csv.loc[:,
                   ['start station id', 'start station name', 'start station latitude', 'start station longitude']]
        # print(sta_info.columns)
        merge_station = unique_stations.merge(sta_info, on='start station id',
                                              how='left').drop_duplicates().reset_index(
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
                            'lon': merge_station['start station longitude'],
                            'lat': merge_station['start station latitude'],
                            'tripcount': len(rows), 'tripdurtion_sum(mins)': rows['tripduration'].sum() / 60,
                            'tripdurtion_mean(mins)': rows['tripduration'].mean() / 60}, ignore_index=True)

        t_df = pd.DataFrame()
        active_station_count = []
        for date in ori_csv['starttime'].unique():
            rows = ori_csv[ori_csv['starttime'] == date]
            count = 0
            for station in rows['start station id'].unique():
                count = rows[rows['start station id'] == station]['start station id'].count()
                active_station_count.append(count)
            t_df = df.append({'date': date, 'active_stations_id': rows['start station id'].unique(),
                              'active_stations_name': rows['start station name'].unique(),
                              'active_stations_count': active_station_count,
                              'lon': merge_station['start station longitude'],
                              'lat': merge_station['start station latitude'],
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
        t_df = gpd.GeoDataFrame(df, geometry='geometry', crs={'init': 'epsg:4326'})

        return t_df

    def complex_drop(self):

        t_df = self.complex_csv()

        t_df.drop(['active_stations_id', 'active_stations_name', 'active_stations_count', 'lon', 'lat'], axis=1,
                  inplace=True)

        return t_df

