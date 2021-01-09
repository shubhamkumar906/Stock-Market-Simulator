from flask import Flask, redirect, url_for, render_template, request




############## STOCK PRICE PREDICTION StockMarketSimulator Part-1
#########################################
def price_prediction(stock_name,forecast_out):
    import quandl
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split

    df = quandl.get("WIKI/"+stock_name)
    #print(df.head())
    df = df[['Adj. Close']]
    #print(df.head())
    # A variable for predicting 'n' days out into the future
    forecast_out = int(forecast_out)
    # Create another column (the target or dependent variable) shifted 'n' units up
    df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
    #print(df.head())
    #print(df.tail())
    ### Creating the independent dataset(X)
    # Converitng the dataframe to a numpy array
    x = np.array(df.drop(['Prediction'],1))
    # Removing the last 'n' rows
    x = x[:-forecast_out]
    #print(x)
    ### Creating the dependent dataset (Y)
    # Converting the dataframe into a numpy array (All the values including the NaN's)
    y = np.array(df['Prediction'])
    # Get all of the y values except the last 'n' rows
    y = y[:-forecast_out]
    #print(y)
    # Splitting the dataset into 80% training and 20% testing
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    # Creating and training the model i.e. Support Vector Machine (Regressor)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train,y_train)
    # Testing Model : Score returns the coefficient of determination R^2 of the prediction.
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)
    #print('svm_confidence:',svm_confidence)
    # Create and train the Linear Regression Model
    lr = LinearRegression()
    # Training the model
    lr.fit(x_train, y_train)
    # Testing Model : Score returns the coefficient of determination R^2 of the prediction.
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)
    #print('lr_confidence:',lr_confidence)
    # Set a variable x_forecast equal to the last 30 rows of the original data set from Adj. Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    #print(x_forecast)
    # Print the Linear Regression Model predictions for next 'n' days
    lr_prediction = lr.predict(x_forecast)
    #print(lr_prediction)
    # Print the Support Vector Model predictions for next 'n' days
    svm_prediction = svr_rbf.predict(x_forecast)
    #print(svm_prediction)
    r = svm_prediction
    return (r)

############## PRICE PREDICTION ends
##############################################




############## STOCK PREDICTION, SIMPLE VOLUME ANALYSIS StockMarketSimulator Part-2
#########################################

def stock_prediction():
    import time
    import yfinance as yf
    import pandas as pd
    df = pd.read_csv('companylist.csv')
    #print(df['Symbol'])
    increased_symbols = []
    t_end = time.time() + 30

    for stock in df['Symbol']:
        while time.time() < t_end:
            stock = stock.upper()
            if '^' in stock:
                pass
            else:
                try:
                  stock_info = yf.Ticker(stock)
                  hist = stock_info.history(period='5d')
                  previous_averaged_volume = hist['Volume'].iloc[1:4:1].mean()
                  todays_volume = hist['Volume'][-1]
                  if todays_volume > previous_averaged_volume * 4:
                    increased_symbols.append(stock)
                except:
                  pass
                print(increased_symbols)
        return(str(increased_symbols))
############## STOCK PREDICTION, SIMPLE VOLUME ANALYSIS ends
##############################################



############## ALERTS AND NOTIFICATIONS, ALERTS BASED ON THRESHOLDS StockMarketSimulator Part-3
#########################################
def stock_alert(symbol,thres):
    import pandas as pd
    from alpha_vantage.timeseries import TimeSeries
    import time

    api_key = '67O5YS2MXMMVBHVL'

    ts = TimeSeries(key=api_key, output_format='pandas')

    data, meta_data = ts.get_intraday(symbol, interval='1min', outputsize='full')
    #print(data)

    close_data = data['4. close']
    percentage_change = close_data.pct_change()

    #print(percentage_change)

    last_change = percentage_change[-1]
    if abs(last_change) > float(thres):
        alt = symbol + " STOCK ALERT !!!    " + str(last_change)
        # print('Microsoft Alert:' + str(last_change))
        return alt
    else:
        return("No alerts for this stock, check after sometime")
############## ALERTS AND NOTIFICATIONS, ALERTS BASED ON THRESHOLDS ends
##############################################



############## COMPARISON_STOCKS, SIMPLE MOVING AVERAGE StockMarketSimulator Part-4
#########################################
def stock_comparison(symbol,interval):
    import pandas as pd
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.techindicators import TechIndicators
    import matplotlib.pyplot as plt
    api_key = '67O5YS2MXMMVBHVL'

    ts = TimeSeries(key = api_key, output_format='pandas')
    data_ts, meta_data = ts.get_intraday(symbol, interval, outputsize = 'full')
    #print(data_ts)

    period = 60
    ti = TechIndicators(key=api_key, output_format='pandas')
    data_ti, meta_data_ti = ti.get_sma(symbol, interval, time_period=period, series_type='close')
    #print(data_ti)

    df1 = data_ti
    df2 = data_ts['4. close'].iloc[period-1::]
    df2.index = df1.index
    total_df = pd.concat([df1,df2], axis=1)
    #print(total_df)
    total_df.plot()

    r1 = total_df
    return(r1)

############## COMPARISON_STOCKS, SIMPLE MOVING AVERAGE ends
##############################################

############## TECHINCAL INDICATORS StockMarketSimulator Part-5
#########################################
def technical_indicators(symbol,interval):
    import pandas as pd
    from alpha_vantage.techindicators import TechIndicators
    import matplotlib.pyplot as plt
    api_key = '67O5YS2MXMMVBHVL'

    period = 60
    ti = TechIndicators(key=api_key, output_format='pandas')
    data_rsi, meta_data_rsi = ti.get_rsi(symbol, interval, time_period=period, series_type='close')
    data_sma, meta_data_sma = ti.get_sma(symbol, interval, time_period=period, series_type='close')
    #print(data_ti)

    df1 = data_sma.iloc[1::]
    df2 = data_rsi
    df1.index = df2.index

    fig, ax1 = plt.subplots()
    ax1.plot(df1, 'b-')
    ax2 = ax1.twinx()
    ax2.plot(df2, 'r.')
    plt.title("SMA & RSI graph")
    # plt.show()
    r2 = df1,df2
    return (r2)
############## TECHNICAL INDICATORS ends
##############################################








app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/price_prediction.html", methods=["POST", "GET"])
def Stock_Price_Prediction():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        forecast_out = request.form["forecast_out"]
        ans = price_prediction(stock_name,forecast_out)
        return redirect(url_for("user", usr = ans))
    return render_template("price_prediction.html")

@app.route("/stock_prediction.html", methods=["POST", "GET"])
def Stock_Name_Prediction():
    if(request.method == "POST"):
        ans = stock_prediction()
        return redirect(url_for("user", usr = ans))
    return render_template("stock_prediction.html")


@app.route("/alerts.html", methods=["POST", "GET"])
def AlertRealTime():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        thres = request.form["thres"]
        ans = stock_alert(stock_name, thres)
        return redirect(url_for("user", usr = ans))
    return render_template("alerts.html")

@app.route("/comparison_stocks.html", methods=["POST", "GET"])
def Comparison_stocks():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        interval = request.form["interval"]
        ans = stock_comparison(stock_name,interval)
        return redirect(url_for("user", usr = ans))
    return render_template("comparison_stocks.html")

@app.route("/technical_indicators.html", methods=["POST", "GET"])
def Technical_Indicators():
    if(request.method == "POST"):
        stock_name = request.form["stock_name"]
        interval = request.form["interval"]
        ans = technical_indicators(stock_name,interval)
        return redirect(url_for("user", usr = ans))
    return render_template("technical_indicators.html")

@app.route("/index.html", methods=["POST", "GET"])
def Home():
    return render_template("index.html")

@app.route("/login.html", methods=["POST", "GET"])
def Login():
    return render_template("login.html")

@app.route("/signup.html", methods=["POST", "GET"])
def SignUp():
    return render_template("signup.html")

@app.route("/about_us.html", methods=["POST", "GET"])
def AboutUs():
    return render_template("about_us.html")

@app.route("/contact_us.html", methods=["POST", "GET"])
def ContactUs():
    return render_template("contact_us.html")



@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"


if(__name__=="__main__"):
    app.run(debug=True)

