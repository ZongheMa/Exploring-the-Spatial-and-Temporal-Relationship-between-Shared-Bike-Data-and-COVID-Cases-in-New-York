from ESDA import *

# # Compute Moran's I statistic to test for spatial autocorrelation in COVID-19 cases and bikes usage
# w = pysal.lib.weights.Queen.from_dataframe(bikes)
# mi_covid = pysal.explore.esda.moran.Moran(bikes['covid_cases'], w)
# mi_bikes = pysal.explore.esda.moran.Moran(bikes['bikes'], w)
