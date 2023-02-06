import numpy as np
import pandas as pd
import os

original_covid_nyc = pd.read_csv('https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv')
covid_nyc = pd.DataFrame(original_covid_nyc,
                         columns=['date_of_interest', 'CASE_COUNT', 'HOSPITALIZED_COUNT', 'DEATH_COUNT',
                                  'BX_CASE_COUNT', 'BX_DEATH_COUNT', 'Bk_CASE_COUNT', 'Bk_DEATH_COUNT', 'MN_CASE_COUNT',
                                  'MN_DEATH_COUNT', 'QN_CASE_COUNT', 'QN_DEATH_COUNT','SI_CASE_COUNT', 'SI_DEATH_COUNT'])
covid_nyc = covid_nyc.loc[(covid_nyc['date_of_interest'] >= '03/01/2020') & (covid_nyc['date_of_interest'] <= '12/31/2022')]


print(original_covid_nyc.head())
print(covid_nyc)
