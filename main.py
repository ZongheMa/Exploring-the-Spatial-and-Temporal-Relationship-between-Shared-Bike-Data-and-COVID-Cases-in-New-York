from data_clean import *
import time
from tqdm import tqdm


# Shared bike DataSets
# url_nyc = 'https://s3.amazonaws.com/tripdata/index.html'
# url_boston = 'https://divvy-tripdata.s3.amazonaws.com/index.html'
# url_chicago = 'https://s3.amazonaws.com/hubway-data/index.html'

# NYC Covid DataSets
# download_link('https://s3.amazonaws.com/tripdata/201306-citibike-tripdata.zip')

# Unzip raw files
# source_folder = '/Users/zonghe/Downloads/'
# target_folder = '/Users/zonghe/Documents/Modules/Term2/CEGE0042_STDM/STDM/data/shared bike datasets/ori_nyc'
# unzip_files(source_folder, target_folder)

# Data cleaning
csv_files = get_csv_paths('data/shared bike datasets/ori_nyc')
for i in tqdm(range(len(csv_files))):
    # df = NYUbike_Clean(csv_files[i]).uniform() # uniform the column names of the csv files
    print(f'\nThe {i+1}/{len(csv_files)} file is being processed...\n')
    df = NYUbike_Clean(csv_files[i]).basic_csv() # basic cleaning
    # save as geojson
    df.to_file('data/shared bike datasets/nyc/' + 'cleaned-' + os.path.splitext(os.path.basename(csv_files[i]))[0] + '.geojson', driver='GeoJSON')
print('NYC shared bike data cleaning completed!')
