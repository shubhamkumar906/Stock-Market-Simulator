pip install quandl
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
df = quandl.get("WIKI/FB")
print(df.head())
df = df[['Adj. Close']]
print(df.head())
# A variable for predicting 'n' days out into the future
forecast_out = 30
# Create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
print(df.head())
print(df.tail())
### Creating the independent dataset(X)
# Converitng the dataframe to a numpy array
x = np.array(df.drop(['Prediction'],1))
# Removing the last 'n' rows
x = x[:-forecast_out]
print(x)
### Creating the dependent dataset (Y)
# Converting the dataframe into a numpy array (All the values including the NaN's)
y = np.array(df['Prediction'])
# Get all of the y values except the last 'n' rows
y = y[:-forecast_out]
print(y)
# Splitting the dataset into 80% training and 20% testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# Creating and training the model i.e. Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train,y_train)
# Testing Model : Score returns the coefficient of determination R^2 of the prediction.
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print('svm_confidence:',svm_confidence)
# Create and train the Linear Regression Model
lr = LinearRegression()
# Training the model
lr.fit(x_train, y_train)
# Testing Model : Score returns the coefficient of determination R^2 of the prediction.
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print('lr_confidence:',lr_confidence)
# Set a variable x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)
# Print the Linear Regression Model predictions for next 'n' days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)
# Print the Support Vector Model predictions for next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)