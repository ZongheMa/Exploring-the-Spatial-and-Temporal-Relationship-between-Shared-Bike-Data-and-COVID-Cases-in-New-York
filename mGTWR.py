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
                         geometry=gpd.points_from_xy(bikes['start station longitude'], bikes['start station latitude']))
bikes = bikes.set_crs(epsg=4326, allow_override=True)
gdf = gpd.GeoDataFrame()
gdf['covid_cases'] = np.nan
count = 0
for i in tqdm(bikes['date'].unique()):
    if count == 30:
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


print(gdf.columns)
coords = gdf[['start station latitude', 'start station longitude']].to_csv('data/SharedBike/coords.csv')
# coords = list(zip(gdf['start station longitude'], gdf['start station latitude']))

X = gdf[
    ['covid_cases', 'trip count', 'tripduration_sum(mins)', 'tripduration_mean(mins)', 'usertype_member_count']].astype(
    'float16').to_csv('data/SharedBike/X.csv')
y = gdf[['trip count']].astype('float16').to_csv('data/SharedBike/y.csv')

# # Normalize the variables
# min_max_scaler = preprocessing.MinMaxScaler()
# X[X.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(X.select_dtypes(include=[np.number]))
# y[y.select_dtypes(include=[np.number]).columns] = min_max_scaler.fit_transform(y.select_dtypes(include=[np.number]))

t = gdf['date'].to_csv('data/SharedBike/t.csv')
float_col = []
for date_str in t:
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.timestamp()
    float_val = float(timestamp)
    float_col.append(float_val)


sel_multi = SearchMGTWRParameter(coords, float_col, X, y, kernel='gaussian', fixed=True)
bws = sel_multi.search()
# bws = sel_multi.search(multi_bw_min=[0.1], verbose=True, tol_multi=1.0e-4, time_cost=True)

mgtwr = MGTWR(coords, float_col, X, y, sel_multi, kernel='gaussian', fixed=True).fit()

print(mgtwr.R2)