import itertools

from data_clean import *
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf, pacf


# ESTDA part
df = merge_csv('data/SharedBike/nyc')
# print(bike.iloc[0])
# print(bike.head())
# print(bike.describe())

agg_func = {'start station id': 'count', 'trip count': 'sum', 'tripduration_sum(sec)': 'sum',
            'tripduration_mean(sec)': 'sum', 'tripduration_sum(mins)': 'sum', 'tripduration_mean(mins)': 'sum',
            'CASE_COUNT': 'sum', 'HOSPITALIZED_COUNT': 'sum', 'DEATH_COUNT': 'sum',
            'BX_CASE_COUNT': 'sum', 'BX_DEATH_COUNT': 'sum', 'BK_CASE_COUNT': 'sum', 'BK_DEATH_COUNT': 'sum',
            'MN_CASE_COUNT': 'sum', 'MN_DEATH_COUNT': 'sum', 'QN_CASE_COUNT': 'sum', 'QN_DEATH_COUNT': 'sum',
            'SI_CASE_COUNT': 'sum', 'SI_DEATH_COUNT': 'sum'}
df_temporal = df.groupby('date').agg(agg_func)
df_temporal.index = pd.to_datetime(df_temporal.index)
print('df_temporal')
print(df_temporal.shape)
print(df_temporal.head())
print(df_temporal.iloc[0])
print(df_temporal.describe())
print(df_temporal.columns)

# visualize the temporal data
# sns.lineplot(data=df_temporal, x='date', y='tripduration_sum(mins)')
# cmap_r = cm.get_cmap('Reds_r', 3)
# cmap_b = cm.get_cmap('Blues_r', 3)
# cmap_o = cm.get_cmap('ocean', len(df_temporal.columns))
cmap_r = ['maroon', 'brown', 'crimson']
cmap_b = ['blue', 'darkblue', 'lightblue']
fig, ax1 = plt.subplots(figsize=(18, 6), dpi=300)
ax2 = ax1.twinx()
count_r = 0
count_b = 0
for i in range(len(df_temporal.columns)):
    if df_temporal.columns[i] in ['BX_CASE_COUNT', 'BX_DEATH_COUNT', 'BK_CASE_COUNT',
                                  'BK_DEATH_COUNT', 'MN_CASE_COUNT', 'MN_DEATH_COUNT', 'QN_CASE_COUNT',
                                  'QN_DEATH_COUNT', 'SI_CASE_COUNT', 'SI_DEATH_COUNT']:
        ax1.plot(df_temporal.index, df_temporal.iloc[:, i], color='lightgrey', label=df_temporal.columns[i],
                 linewidth=0.6)
    elif df_temporal.columns[i] in ['CASE_COUNT', 'HOSPITALIZED_COUNT', 'DEATH_COUNT']:

        ax1.plot(df_temporal.index, df_temporal.iloc[:, i], color=cmap_r[count_r], label=df_temporal.columns[i],
                 linewidth=0.6)
        count_r += 1

    elif df_temporal.columns[i] in ['tripduration_sum(mins)', 'tripduration_mean(mins)', 'trip count']:
        ax2.plot(df_temporal.index, df_temporal.iloc[:, i], color=cmap_b[count_b], label=df_temporal.columns[i],
                 linewidth=0.6)
        count_b += 1

ax1.set_xlabel('Date')
ax1.set_ylabel('Covid-related Values', color='darkblue')
ax1.legend(loc='upper left')
ax1.tick_params(axis='y', labelcolor='darkblue')

ax2.set_ylabel('Shared Bike Values', color='maroon', loc='top')
ax2.legend(loc='upper right')
ax2.tick_params(axis='y', labelcolor='maroon')

# set the x-axis ticks
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# # set different x-axis limits
# ax1.set_xlim(df_temporal.index[0], pd.Timestamp('2019-10-31'))
# ax2.set_xlim(pd.Timestamp('2021-11-01'), df_temporal.index[-1])
# ax1.xaxis.set_tick_params(which='major', pad=15)
# ax2.xaxis.set_tick_params(which='major', pad=15)

# rotate the x-axis ticks
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

'''# # plt.plot(df_temporal.drop('tripduration_sum(sec)', axis=1), linewidth=0.6)
#
# # plt.scatter(df_temporal.index, df_temporal.values, c=df_temporal.values, cmap='cool')
# plt.legend(df_temporal.drop(['tripduration_sum(sec)', 'tripduration_mean(sec)'], axis=1).columns)
# plt.title('Temporal data in NYC')
# plt.xlabel('Date')
# plt.ylabel('Value')
#
# plt.show()
#
# df_v = df_temporal['tripduration_sum(mins)']
# fig, ax = plt.subplots(2, 1, figsize=(10, 8))
# sm.graphics.tsa.plot_acf(df_v, lags=10, ax=ax[0])
# sm.graphics.tsa.plot_pacf(df_v, lags=10, ax=ax[1])
# plt.show()'''

# histogram and density plot
fig, ax = plt.subplots(3, 1, figsize=(10, 5))
sns.distplot(df_temporal['tripduration_sum(mins)'], hist=True, kde=True, bins=30, ax=ax[0])
sns.distplot(df_temporal['trip count'], hist=True, kde=True, bins=30, ax=ax[1])
sns.distplot(df_temporal['CASE_COUNT'], hist=True, kde=True, bins=30, ax=ax[2])
plt.subplots_adjust(hspace=0.5)
plt.title('Histogram and Density Plot')
plt.show()

plt.figure(figsize=(10, 5))
sns.distplot(df_temporal, hist=True, kde=True, bins=30)
plt.title('Histogram and Density Plot of the temporal dataset')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend(['Histogram', 'Density'])
plt.show()

# decomposition the time series
decomposition = seasonal_decompose(df_temporal['tripduration_sum(mins)'], model='additive', period=365)
# adjust the period to 365 days and 12 months for diferent analysis


# visualize the time series after decomposition
plt.figure(figsize=(10, 10))
plt.title('Decomposition of the tripduration_sum(mins) for NYC from 2019-01-01 to 2022-12-31, period is {}')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(4, 1, 1)
plt.plot(df_temporal['tripduration_sum(mins)'], label='Original', linewidth=0.6)
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonality', linewidth=0.6)
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(residual, label='Residuals', linewidth=0.6)
plt.legend(loc='upper left')
plt.subplots_adjust(hspace=0.4)
plt.show()

# ARIMA model
print('\n{:-^60s}'.format('ARIMA Model'))

# check the stationarity of the time series using ADF test

result_1 = adfuller(df_temporal['tripduration_sum(mins)'], autolag='BIC')
print('p-value: %f' % result_1[1])
print('ADF Statistic: %f' % result_1[0])
print('Critical Values:')
for key, value in result_1[4].items():
    print('\t%s: %.3f' % (key, value))

# difference the time series
diff_1 = df_temporal['tripduration_sum(mins)'].diff().dropna()

# plot the time series after differencing
plt.figure(figsize=(15, 5))
plt.plot(diff_1, linewidth=0.5, color='red')
plt.title('Differenced time series(order=1)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# check the stationarity of the differenced time series using ADF test
print('\nDifference order 1:')
result_1 = adfuller(diff_1)
print('p-value: %f' % result_1[1])
print('ADF Statistic: %f' % result_1[0])
print('Critical Values:')
for key, value in result_1[4].items():
    print('\t%s: %.3f' % (key, value))

# choose the parameters of ARIMA model
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
sm.graphics.tsa.plot_acf(diff_1, ax=ax[0])
sm.graphics.tsa.plot_pacf(diff_1, ax=ax[1])
plt.title('ACF and PACF of differenced time series(order=1)')
plt.show()

# fit the ARIMA model
model = ARIMA(df_temporal['tripduration_sum(mins)'], order=(1, 1, 1))
model_fit = model.fit()

# output the summary of the model
print(model_fit.summary())

# visualize the residuals of the ARIMA model
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(linewidth=0.5, color='green')
plt.legend(loc='upper right')
plt.title('Residuals of ARIMA model')
plt.show()

# forecast the time series in ARIMA model
forecast = model_fit.forecast(steps=365)
print(forecast)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
sm.graphics.tsa.plot_acf(residuals, ax=ax[0])
sm.graphics.tsa.plot_pacf(residuals, ax=ax[1])
plt.title('ACF and PACF of ARIMA model residuals')
plt.show()

# SARIMA model
print('\n{:-^60s}'.format('SARIMA Model'))

# define the p, d, and q parameters
p = d = q = range(0, 2)

# generate all different combinations of p, d, and q triplets
pdq = list(itertools.product(p, d, q))

# generate all different combinations of seasonal p, d, and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

# find the best parameters for the model
best_bic = np.inf
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df_temporal['tripduration_sum(mins)'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            if results.bic < best_bic:
                best_bic = results.bic
                best_params = param
                best_params_seasonal = param_seasonal
        except:
            continue

print('Best SARIMA parameters:', best_params, best_params_seasonal)

# fit the SARIMA model
model = sm.tsa.statespace.SARIMAX(df_temporal['tripduration_sum(mins)'],
                                  order=best_params,
                                  seasonal_order=best_params_seasonal,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
model_fit = model.fit()

# output the SARIMA model summary
print(model_fit.summary())

# plot the residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(linewidth=0.5, color='green')
plt.legend(loc='upper right')
plt.title('Residuals of SARIMA model')
plt.show()

# forecast future data
forecast = model_fit.forecast(steps=365)
print(forecast)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
sm.graphics.tsa.plot_acf(residuals, ax=ax[0])
sm.graphics.tsa.plot_pacf(residuals, ax=ax[1])
plt.title('ACF and PACF of SARIMA model residuals')
plt.show()
