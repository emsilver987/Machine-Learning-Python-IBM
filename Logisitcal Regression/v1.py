import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Load in URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
churn_df

# Declaring Fields
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df

# Setting Input Fields and Target Fields 
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]  #print the first 5 values
y = np.asarray(churn_df['churn'])
y[0:5] #print the first 5 values

# Normalize Data
X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]

# Split Training and Test Size
X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

# SciKit-Learn Package
LR = LogisticRegression().fit(X_train,y_train)

# Predict Churn Variable from Training Set
yhat = LR.predict(X_test)
yhat[:10]

# Predicition Probability - Threshold .5
yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]

# Looking at how the input values affect our churn model
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()