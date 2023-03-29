import numpy as np
import pandas as pd
import geopandas as gpd
import pysal as ps
import matplotlib.pyplot as plt
import libpysal as lp
from libpysal import weights
from pysal.explore import esda

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#
df = pd.read_csv('data/export data for spatial analysis/df_GWR.csv')

df['geometry'] = gpd.points_from_xy(df['start station longitude'], df['start station latitude'])
df = gpd.GeoDataFrame(df, geometry='geometry')
print(df.columns)
shp = gpd.read_file('data/Geography-resources/MODZCTA_2010.shp')
print(shp.columns)
shp.plot()
plt.show()

# w = ps.lib.weights.Queen.from_dataframe(df)  # create spatial weights matrix
# mi_bike = ps.explore.esda.Moran(df['trip count'], w)  # calculate Moran's I for bike trips
# mi_covid = ps.explore.esda.Moran(df['case count'], w)  # calculate Moran's I for covid cases


# plot 3D surface plot example
# # Create some sample data
# x = np.linspace(-5, 5, 50) # x coordinates
# y = np.linspace(-5, 5, 50) # y coordinates
# X, Y = np.meshgrid(x, y) # create a grid of x and y values
# Z = np.sin(np.sqrt(X**2 + Y**2)) # z values as a function of x and y
#
# # Create a 3D axes object
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# # Plot a surface plot of the data
# ax.plot_surface(X, Y, Z, cmap='viridis')
#
# # Add labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('Surface plot of z = sin(sqrt(x^2 + y^2))')

# # 提取shp文件的x和y坐标
# x = np.array(nyc.geometry.centroid.x)
# y = np.array(nyc.geometry.centroid.y)
#
# # 选择一个列名作为z轴
# z = np.array(nyc['2022-05-25'])
#
# # 使用colormap参数将各个列的值映射为颜色
# cmap = plt.cm.get_cmap("viridis")
#
# # 或者使用facecolors参数指定每个面的颜色
# # facecolors = ["red", "green", "blue", ...]
#
# # 创建一个3D坐标轴对象
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# # 绘制每个shp文件的三角形曲面图
# ax.plot_trisurf(x, y, z, cmap=cmap)
#
# # 或者使用facecolors参数指定每个面的颜色
# # ax.plot_trisurf(x, y, z, facecolors=facecolors)
#
# # 添加标签和标题
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
#
# plt.show()


# Moran's I
w = weights.Queen.from_dataframe(df)
print(w.summary())

moran = esda.Moran(df['trip count'], w)

# 打印莫兰指数的摘要信息
print(moran.summary())