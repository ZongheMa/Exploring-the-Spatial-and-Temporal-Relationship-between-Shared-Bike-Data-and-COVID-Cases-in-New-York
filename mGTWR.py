import geopandas as gpd
import numpy.linalg
import pandas as pd
import numpy as np
from data_clean import merge_csv
from tqdm import tqdm
from mgtwr.sel import SearchMGTWRParameter
from mgtwr.model import MGTWR
import datetime
from sklearn import preprocessing

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    import warnings

    warnings.filterwarnings("ignore")

    nyc = gpd.read_file('data/geo/Geography-resources/MODZCTA_2010.shp')
    nyc['MODZCTA'] = nyc['MODZCTA'].astype('int64')
    nyc.to_crs(epsg=4326, inplace=True)
    dfzip = pd.read_csv('data/Covid_cases/covid_nyc_byzipcode_T.csv', index_col=0, dtype={'MODZCTA': 'int16'})
    covid_byzip = pd.merge(nyc, dfzip, on='MODZCTA', how='left')
    print(covid_byzip.columns)
    # covid_byzip.columns[3:] = pd.to_datetime(covid_byzip.columns[3:], format='%Y-%m-%d')

    bikes = merge_csv('data/SharedBike/nyc', usecols=['date', 'start station id', 'start station name',
                                                      'start station latitude', 'start station longitude',
                                                      'trip count', 'tripduration_sum(mins)',
                                                      'tripduration_mean(mins)', 'usertype_member_count',
                                                      'usertype_casual_count'])
    bikes = gpd.GeoDataFrame(bikes,
                             geometry=gpd.points_from_xy(bikes['start station longitude'],
                                                         bikes['start station latitude']))
    bikes = bikes.set_crs(epsg=4326, allow_override=True)
    gdf = gpd.GeoDataFrame()
    gdf['covid_cases'] = np.nan
    count = 0
    for i in tqdm(bikes['date'].unique()):
        if count == 14:
            break

        if i in covid_byzip.columns:
            print(i)
            sjoin = gpd.sjoin(bikes[bikes['date'] == i], covid_byzip[['MODZCTA', i, 'geometry']], how='left',
                              op='intersects')
            sjoin = sjoin[pd.notnull(sjoin[i])]
            sjoin = sjoin.drop(columns=['index_right'])
            gdf = pd.concat([gdf, sjoin])
            gdf.loc[bikes['date'] == i, 'covid_cases'] = sjoin[i].values
            gdf.drop(columns=i, inplace=True)
            count += 1
            print(count)
        else:
            continue

    df = gdf.drop(columns=['geometry', 'MODZCTA', 'usertype_casual_count'], axis=1)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)

    df = df.groupby('start station id').resample('W').agg(
        {'start station name': 'first', 'start station latitude': 'first', 'start station longitude': 'first',
         'covid_cases': 'sum', 'trip count': 'sum', 'tripduration_sum(mins)': 'sum', 'tripduration_mean(mins)': 'sum',
         'usertype_member_count': 'sum'})
    df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10 ** 9
    df.to_csv('data/df_M.csv')

    coords = df[['start station latitude', 'start station longitude']]
    # coords = list(zip(gdf['start station longitude'], gdf['start station latitude']))

    X = df[
        ['covid_cases', 'tripduration_sum(mins)', 'tripduration_mean(mins)', 'usertype_member_count']].astype(
        'float16')
    y = df[['trip count']].astype(int)

    # # Normalize the variables
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X[X.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(X.select_dtypes(include=[np.number]))
    # y[y.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(y.select_dtypes(include=[np.number]))

    print('got it')

    t = df[['timestamp']].astype(int)

    # int_col = []
    # for date in t:
    #     # date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    #     timestamp = date.timestamp()
    #     int_val = int(timestamp)
    #     int_col.append(int_val)
    #
    # t = np.array(int_col)
    # t = t.reshape(-1, 1)
    # print('after t parse')
    # #
    # sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
    # bws = sel_multi.search()

    print('fit')


    class sel_multi:
        def __init__(self, bws):
            self.bws = bws


    Bws = sel_multi([0.9, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    selector = sel_multi(Bws.bws)
    mgtwr = MGTWR(coords, t, X, y, selector, kernel='gaussian', fixed=True).fit()

    print(mgtwr.R2)
