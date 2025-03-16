import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)  # load dataset
print("Loading 5 Random Samples\n", df.sample(5))

print("Loading Statstical Analysis of Data\n", df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 
# print("Selected a sample and features\n", cdf.sample(9))

# viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print("Visualize Histogram")
# viz.hist()

# print("Show plot")
# plt.show()

# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color = 'orange')
plt.xlabel("Cylinder")
plt.ylabel("Emission")
plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.xlim(0,27)
# plt.show()