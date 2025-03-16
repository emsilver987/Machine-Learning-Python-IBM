import pandas as pd
import matplotlib as plt
import sys
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)  # load dataset
print("Loading 5 Random Samples", df.sample(5))

print("Loading Statstical Analysis of Data", df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 
print("Selected a sample and features", cdf.sample(9))

viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print("Visualize Histogram")
viz.hist()
print("Show plot")
plt.show()