import geopandas as gpd
import pandas as pd
import numpy as np
from data_clean import merge_csv
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from mgtwr.model import MGTWR
from mgtwr.sel import SearchMGTWRParameter
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

if __name__ == '__main__':
    # address issues with multiprocessing on windows and ignore warnings
    from multiprocessing import freeze_support

    freeze_support()
    import warnings

    warnings.filterwarnings("ignore")

    # spatial join and extract covid cases to each station

    # prepare data
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

    # spatial join
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

    # organize df for modelling
    df = gdf.drop(columns=['geometry', 'MODZCTA', 'usertype_casual_count'], axis=1)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)

    df = df.groupby('start station id').resample('W').agg(
        {'start station name': 'first', 'start station latitude': 'first', 'start station longitude': 'first',
         'covid_cases': 'sum', 'trip count': 'sum', 'tripduration_sum(mins)': 'sum', 'tripduration_mean(mins)': 'sum',
         'usertype_member_count': 'sum'})
    df = df.reset_index()

    # df = pd.read_csv('data/df for ST analysis/df_W.csv', index_col=0)
    # check the missing values
    null_cols = df.columns[df.isnull().any()].tolist()
    null_rows = df.isnull().any(axis=1)
    # drop missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    # check the missing values results
    print(df.isnull().sum())
    print(df.isfinite().sum())
    # build int timestamp to meet the model requirement
    df['timestamp'] = pd.to_datetime(df['date']).astype(int) // 10 ** 9
    print(df.shape)
    # df.to_csv('data/df_M.csv')

    coords = df[['start station latitude', 'start station longitude']]
    # coords = list(zip(gdf['start station longitude'], gdf['start station latitude']))

    X = df[['covid_cases', 'tripduration_sum(mins)', 'tripduration_mean(mins)', 'usertype_member_count']].astype(
        'float16')  # using float16 to save memory
    y = df[['trip count']].astype(int)
    t = df[['timestamp']].astype(int)

    # Normalize the variables
    min_max_scaler = MinMaxScaler()
    X[X.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(X.select_dtypes(include=[np.number]))
    y[y.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(y.select_dtypes(include=[np.number]))

    # Select bandwidths
    sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
    bws = sel_multi.search()

    # Fit MGTWR model
    mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel='gaussian', fixed=True).fit()
    print(mgtwr.R2)

    # fit GWR model from mgwr package
    df = df.groupby('start station id').agg(
        {'start station name': 'first', 'start station latitude': 'first', 'start station longitude': 'first',
         'covid_cases': 'sum', 'trip count': 'sum', 'tripduration_sum(mins)': 'sum', 'tripduration_mean(mins)': 'sum',
         'usertype_member_count': 'sum'})
    df = df.reset_index()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    df['geometry'] = gpd.points_from_xy(df['start station longitude'], df['start station latitude'])
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.to_csv('data/df_GWR.csv')
    X = df[['covid_cases', 'tripduration_sum(mins)', 'tripduration_mean(mins)', 'usertype_member_count']]
    y = df[['trip count']]

    sel = Sel_BW(X, y, df.geometry)
    bw = sel.search()

    model = MGWR(df.geometry, y, bw, fixed=True).fit()
    print(model.R2)
    print(model.summary())
