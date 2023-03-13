import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_clean import BikeDataClean
from qgis.core import *

# place_name = "New York City, New York, USA"
# G = ox.graph_from_place(place_name, network_type='all')
#
# ox.save_graph_shapefile(G, 'nyc.shp')
# ox.save_graphml(G, 'nyc.graphml')




nyc = gpd.read_file()
nyc.plot()
plt.show()
print(nyc.head())
print(nyc.columns)
print(nyc.shape)
print(nyc.crs)
print(nyc.info())