import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time

api_key = '67O5YS2MXMMVBHVL'
ts = TimeSeries(key=api_key, output_format='pandas')
def stock_alert(symbol,thres):
    data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
    #print(data)

    close_data = data['4. close']
    percentage_change = close_data.pct_change()

    #print(percentage_change)

    last_change = percentage_change[-1]
    if abs(last_change) > float(thres):
        alt = symbol + " STOCK ALERT !!!  ->  " + str(last_change)
        # print('Microsoft Alert:' + str(last_change))
        return alt
    else:
        return("No alerts for this stock, check after sometime")