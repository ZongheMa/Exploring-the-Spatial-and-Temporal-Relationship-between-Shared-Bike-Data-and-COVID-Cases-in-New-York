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

X = pd.read_csv('data/SharedBike/X.csv', index_col=0)
y = pd.read_csv('data/SharedBike/y.csv', index_col=0)
coords = pd.read_csv('data/SharedBike/coords.csv', index_col=0)
t = pd.read_csv('data/SharedBike/t.csv', index_col=0)

float_col = []

for date_str in t['date']:
    try:
        # 尝试将日期字符串转换为日期对象
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        # 将日期对象转换为UNIX时间戳，并将其转换为浮点数
        timestamp = date_obj.timestamp()
        float_val = float(timestamp)
        # 将浮点数添加到float_col列表中
        float_col.append(float_val)
    except ValueError:
        # 如果日期字符串无法解析为日期对象，或者不是日期字符串，就跳过它
        continue

t = float_col

print("end for")

sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
bws = sel_multi.search()
# bws = sel_multi.search(multi_bw_min=[0.1], verbose=True, tol_multi=1.0e-4, time_cost=True)

mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel='gaussian', fixed=True).fit()

print(mgtwr.R2)