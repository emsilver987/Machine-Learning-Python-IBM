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

# Log Loss - Looking for low number
log_loss(y_test, yhat_prob)


# Exercise 1 - :et us assume we add the feature 'callcard' to the original set of input features. What will the value of log loss be in this case?
url1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn1_df = pd.read_csv(url1)
churn1_df
churn1_df = churn1_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard','churn']]
churn1_df['churn'] = churn1_df['churn'].astype('int')
churn1_df
X1 = np.asarray(churn1_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard']])
X1[0:5]
y1 = np.asarray(churn1_df['churn'])
y1[0:5]
X1_norm = StandardScaler().fit(X1).transform(X1)
X1_norm[0:5]
X1_train, X1_test, y1_train, y1_test = train_test_split( X1_norm, y1, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X1_train,y1_train)
y1hat = LR.predict(X1_test)
y1hat[:10]
y1hat_prob = LR.predict_proba(X1_test)
y1hat_prob[:10]
coefficients = pd.Series(LR.coef_[0], index=churn1_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
print(log_loss(y1_test, y1hat_prob))


# Exercise 2 - Let us assume we add the feature 'wireless' to the original set of input features. What will the value of log loss be in this case?
url2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn2_df = pd.read_csv(url2)
churn2_df
churn2_df = churn2_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'wireless', 'churn']]
churn2_df['churn'] = churn2_df['churn'].astype('int')
churn2_df
X2 = np.asarray(churn2_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'wireless', 'equip']])
X2[0:5]
y2 = np.asarray(churn2_df['churn'])
y2[0:5]
X2_norm = StandardScaler().fit(X2).transform(X2)
X2_norm[0:5]
X2_train, X2_test, y2_train, y2_test = train_test_split( X2_norm, y2, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X2_train,y2_train)
y2hat = LR.predict(X2_test)
y2hat[:10]
y2hat_prob = LR.predict_proba(X2_test)
y2hat_prob[:10]
coefficients = pd.Series(LR.coef_[0], index=churn2_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
print(log_loss(y2_test, y2hat_prob))


# Exercise 3 - What happens to the log loss value if we add both "callcard" and "wireless" to the input features?
url3 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn3_df = pd.read_csv(url3)
churn3_df
churn3_df = churn3_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
churn3_df['churn'] = churn3_df['churn'].astype('int')
churn3_df
X3 = np.asarray(churn3_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'callcard', 'wireless', 'equip']])
X3[0:5]
y3 = np.asarray(churn3_df['churn'])
y3[0:5]
X3_norm = StandardScaler().fit(X3).transform(X3)
X3_norm[0:5]
X3_train, X3_test, y3_train, y3_test = train_test_split( X3_norm, y3, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X3_train,y3_train)
y3hat = LR.predict(X3_test)
y3hat[:10]
y3hat_prob = LR.predict_proba(X3_test)
y3hat_prob[:10]
coefficients = pd.Series(LR.coef_[0], index=churn3_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
print(log_loss(y3_test, y3hat_prob))

# Exercise 4 - What happens to the log loss if we remove the feature 'equip' from the original set of input features?
url4 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn4_df = pd.read_csv(url4)
churn4_df
churn4_df = churn4_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'churn']]
churn4_df['churn'] = churn4_df['churn'].astype('int')
churn4_df
X4 = np.asarray(churn4_df[['tenure', 'age', 'address', 'income', 'ed', 'employ']])
X4[0:5]
y4 = np.asarray(churn4_df['churn'])
y4[0:5]
X4_norm = StandardScaler().fit(X4).transform(X4)
X4_norm[0:5]
X4_train, X4_test, y4_train, y4_test = train_test_split( X4_norm, y4, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X4_train,y4_train)
y4hat = LR.predict(X4_test)
y4hat[:10]
y4hat_prob = LR.predict_proba(X4_test)
y4hat_prob[:10]
coefficients = pd.Series(LR.coef_[0], index=churn4_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
print(log_loss(y4_test, y4hat_prob))


# Exercise 5 - What happens to the log loss if we remove the features 'income' and 'employ' from the original set of input features?
url5 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn5_df = pd.read_csv(url5)
churn5_df
churn5_df = churn5_df[['tenure', 'age', 'address', 'income', 'ed', 'equip', 'churn']]
churn5_df['churn'] = churn5_df['churn'].astype('int')
churn5_df
X5 = np.asarray(churn5_df[['tenure', 'age', 'address', 'income', 'ed', 'equip']])
X5[0:5]
y5 = np.asarray(churn5_df['churn'])
y5[0:5]
X5_norm = StandardScaler().fit(X5).transform(X5)
X5_norm[0:5]
X5_train, X5_test, y5_train, y5_test = train_test_split( X5_norm, y5, test_size=0.2, random_state=4)
LR = LogisticRegression().fit(X5_train,y5_train)
y5hat = LR.predict(X5_test)
y5hat[:10]
y5hat_prob = LR.predict_proba(X5_test)
y5hat_prob[:10]
coefficients = pd.Series(LR.coef_[0], index=churn5_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()
print(log_loss(y5_test, y5hat_prob))