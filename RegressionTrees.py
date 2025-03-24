from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

# Read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data

# Plot Correlation Values
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

# Extract Input and Target and Normalize
y = raw_data[['tip_amount']].values.astype('float32')
proc_data = raw_data.drop(['tip_amount'], axis=1)
X = proc_data.values
X = normalize(X, axis=1, norm='l1', copy=False)

# Build Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8, 
                               random_state=35)
dt_reg.fit(X_train, y_train)

# Evaluate 
y_pred = dt_reg.predict(X_test)
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))
r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

# Q1 What if we change the max_depth to 12? How would the $MSE$ and $R^2$ be affected? 
dt_altered = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=12, 
                               random_state=35)
dt_altered.fit(X_train, y_train)
y_pred = dt_altered.predict(X_test)
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))
r2_score = dt_altered.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

# Q2 - Identify the top 3 features with the most effect on the `tip_amount`.
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
abs(correlation_values).sort_values(ascending=False)[:3]

# Q3 - Since we identified 4 features which are not correlated with the target variable, try removing these variables from the input set and see the effect on the $MSE$ and $R^2$ value.
raw_data = raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
y1 = raw_data[['tip_amount']].values.astype('float32')
proc_data = raw_data.drop(['tip_amount'], axis=1)
X1 = proc_data.values
X1 = normalize(X1, axis=1, norm='l1', copy=False)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8, 
                               random_state=35)
dt_reg.fit(X1_train, y1_train)
y1_pred = dt_reg.predict(X1_test)
mse_score = mean_squared_error(y1_test, y1_pred)
print('MSE score dropping 3 least important: {0:.3f}'.format(mse_score))
r2_score = dt_reg.score(X1_test,y1_test)
print('R^2 score dropping 3 least important: {0:.3f}'.format(r2_score))

# Q4 - Check the effect of **decreasing** the `max_depth` parameter to 4 on the $MSE$ and $R^2$ values.
dt_altered = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=4, 
                               random_state=35)
dt_altered.fit(X_train, y_train)
y_pred = dt_altered.predict(X_test)
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))
r2_score = dt_altered.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))
