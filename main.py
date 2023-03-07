from data_clean import *
from sklearn.ensemble import IsolationForest
import pysal

# Data cleaning part
''''# Unzip raw files
source_folder = '/Users/zonghe/Downloads/'
target_folder = '/Users/zonghe/Documents/Modules/Term2/CEGE0042_STDM/STDM/data/SharedBike/ori_bst'
unzip_files(source_folder, target_folder)
'''

'''
# Covid-19 data from NYC Health(https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv)
df = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv')
covid_nyc = pd.DataFrame(df,
                         columns=['date_of_interest', 'CASE_COUNT', 'HOSPITALIZED_COUNT', 'DEATH_COUNT',
                                  'BX_CASE_COUNT', 'BX_DEATH_COUNT', 'BK_CASE_COUNT', 'BK_DEATH_COUNT', 'MN_CASE_COUNT',
                                  'MN_DEATH_COUNT', 'QN_CASE_COUNT', 'QN_DEATH_COUNT', 'SI_CASE_COUNT',
                                  'SI_DEATH_COUNT'])
covid_nyc['date_of_interest'] = pd.to_datetime(covid_nyc['date_of_interest'], format='%m/%d/%Y').astype(str)
covid_nyc = covid_nyc.loc[covid_nyc['date_of_interest'] < '2023-02-01']
covid_nyc.rename(columns={'date_of_interest': 'date'}, inplace=True)

db = pd.DataFrame(pd.date_range(start='2019-01-01', end='2020-02-28', freq='D').astype(str), columns=['date'])

covid_nyc = pd.concat([db, covid_nyc], axis=0, join='outer', ignore_index=True).fillna(0)
covid_nyc.to_csv('data/Covid_cases/covid_nyc.csv', index=False)


# Covid cases data from City of BST(https://www.boston.gov/government/cabinets/boston-public-health-commission/covid-19-boston )
df = pd.read_csv('data/Covid_cases/Boston_COVID-19_NewCases.csv')
df = df.loc[:, ['Category1', 'Value']]
df.rename(columns={'Category1': 'date', 'Value': 'cases'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').astype(str)
db = pd.DataFrame(pd.date_range(start='2019-01-01', end='2020-02-21', freq='D').astype(str), columns=['date'])
df = pd.concat([db, df], axis=0, join='outer', ignore_index=True).fillna(0)
df.index = pd.to_datetime(df['date'])
df.resample('D').mean().interpolate() # interpolate the missing values
df.to_csv('data/Covid_cases/covid_bst.csv', index=False)
'''

'''# Data cleaning
csv_files = get_csv_paths('data/SharedBike/ori_nyc')
covid = pd.read_csv('data/Covid_cases/covid_nyc.csv')
for i in tqdm(range(len(csv_files))):
    df = BikeDataClean(csv_files[i]).uniform() # uniform the column names of the csv files
    df = BikeDataClean(csv_files[i]).basic_csv() # basic cleaning

    df.rename(columns={'starttime': 'date'}, inplace=True)
    df = pd.merge(df, covid, on='date', how='left')

    df.to_csv('data/SharedBike/nyc/' + 'cleaned-' + os.path.splitext(os.path.basename(csv_files[i]))[0] + '.csv', index=False)
print('NYC shared bike data cleaning completed!')
'''

# ESTDA part
# bike_csv = get_csv_paths('data/SharedBike/nyc')
# bike = pd.read_csv(bike_csv[0])
# print(bike.head())
# print(bike.columns)
# print(bike.shape)
# print(bike.dtypes)
# print(bike.describe())

# # Compute Moran's I statistic to test for spatial autocorrelation in COVID-19 cases and bikes usage
# w = pysal.lib.weights.Queen.from_dataframe(bikes)
# mi_covid = pysal.explore.esda.moran.Moran(bikes['covid_cases'], w)
# mi_bikes = pysal.explore.esda.moran.Moran(bikes['bikes'], w)