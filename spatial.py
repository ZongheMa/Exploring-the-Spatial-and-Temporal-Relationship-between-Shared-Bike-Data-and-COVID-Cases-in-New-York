import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_clean import merge_csv
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm
import pysal as ps
from gwr.gwr import GWR
from gwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
from pysal.model import spreg
from pysal.model import *
from pysal.explore import *

nyc = gpd.read_file('data/geo/Geography-resources/MODZCTA_2010.shp')
nyc['MODZCTA'] = nyc['MODZCTA'].astype('int64')
nyc.to_crs(epsg=4326, inplace=True)
dfzip = pd.read_csv('data/Covid_cases/covid_nyc_byzipcode_T.csv', index_col=0, dtype={'MODZCTA': 'int64'})
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
    else:
        continue
    if count == 2:
        break

# # normalize the training data of continuous variables
# df['covid_cases'] = (df['covid_cases'] - df['covid_cases'].mean()) / df['covid_cases'].std()
# df['tripduration_sum(mins)'] = (df['tripduration_sum(mins)'] - df['tripduration_sum(mins)'].mean()) / df['tripduration_sum(mins)'].std()
# df['tripduration_mean(mins)'] = (df['tripduration_mean(mins)'] - df['tripduration_mean(mins)'].mean()) / df['tripduration_mean(mins)'].std()



