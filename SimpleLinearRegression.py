import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)  # load dataset
print("Loading 5 Random Samples\n", df.sample(5))

print("Loading Statstical Analysis of Data\n", df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 

# My Code for Cylinder against C02 Emissions
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color = 'orange')
plt.xlabel("Cylinder")
plt.ylabel("Emission")
plt.show()

# Init first Linear Regression Line
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
type(X_train), np.shape(X_train), np.shape(X_train)

# Second Linear Regression Line
X1 = cdf.FUELCONSUMPTION_COMB.to_numpy()
y1 = cdf.CYLINDERS.to_numpy()
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=.2,random_state=42)
type(X1_train), np.shape(X1_train), np.shape(X1_train)

# Create a model object
regressor = linear_model.LinearRegression()
regr = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)
regr.fit(X1_train.reshape(-1, 1), y1_train)

# Print the coefficients
print ("First Linear Regression Model Coefficents")
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)
print ("\nSecond Linear Regression Line:")
print ('Coefficients: ', regr.coef_[0]) # seond linear regression line
print ('Intercept: ',regr.intercept_)

# Plotting Training Data in Blue and Test Data in Blue
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.scatter(X_test, y_test, color='purple')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, 'r')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

# Plot New Linear Regression Line
plt.scatter(X1_test, y1_test, color='purple')
plt.plot(X1_test, regressor.coef_ * X1_test + regr.intercept_, 'r')
plt.xlabel("Full Consumption")
plt.ylabel("Cylinders")
plt.show()

# First Linear Regression Line Error Stats
print("\nFirst Linear Regression Line Error Statistics")
y_test_ = regressor.predict( X_test.reshape(-1,1))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test_, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))
print("Root mean squared error: %.2f" % root_mean_squared_error(y_test_, y_test))
print("R2-score: %.2f" % r2_score( y_test_, y_test) )

y1_test_ = regr.predict( X1_test.reshape(-1,1))
print("\nSecond Linear Regression Line Error Statistics")
print("Mean absolute error: %.2f" % mean_absolute_error(y1_test_, y1_test))
print("Mean squared error: %.2f" % mean_squared_error(y1_test_, y1_test))
print("Root mean squared error: %.2f" % root_mean_squared_error(y1_test_, y1_test))
print("R2-score: %.2f" % r2_score( y1_test_, y1_test) )