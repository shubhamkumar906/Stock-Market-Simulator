import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
api_key = '67O5YS2MXMMVBHVL'
ts = TimeSeries(key = api_key, output_format='pandas')
data_ts, meta_data = ts.get_intraday(symbol='MSFT', interval = '1min', outputsize = 'full')

#print(data_ts)
period = 60
ti = TechIndicators(key=api_key, output_format='pandas')
data_ti, meta_data_ti = ti.get_sma(symbol="MSFT", interval = '1min', time_period=period, series_type='close')

#print(data_ti)
df1 = data_ti
df2 = data_ts['4. close'].iloc[period-1::]
df2.index = df1.index
total_df = pd.concat([df1,df2], axis=1)
ans = data_ts,data_ti,total_df
#print(total_df)
total_df.plot()
graph = plt.show()