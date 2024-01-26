#Scalecast

# This package is for forecasters who know their models will need to be validated and used to produce forecasts 
# on a future horizon. Maybe you don’t want to spend as much 
# time breaking down the fine details of the model. If that is the case, 
# the basic flow of the code, from importing results, to testing and forecasting, goes like this:

#https://towardsdatascience.com/forecast-with-arima-in-python-more-easily-with-scalecast-35125fc7dc2e
#https://pypi.org/project/SCALECAST/

import pandas as pd
import numpy as np
from scalecast.Forecaster import Forecaster
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(14,7)})

df = pd.read_csv('AirPassengers.csv')
f = Forecaster(y=df['#Passengers'],current_dates=df['Month'])

f.generate_future_dates(12) # 12-month forecast horizon
f.set_test_length(.2) # 20% test set
f.set_estimator('arima') # set arima
f.manual_forecast(call_me='arima1') # forecast with arima

f.plot_test_set(ci=True) # view test results
plt.title('ARIMA Test-Set Performance',size=14)
plt.show()
