import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Load in weather dataset and preprocess
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
print(df.head())
print(df.count)
df = df.dropna()
print(df.info())
df.columns
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()

# Function that maps dates
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Exercise 1. Map the dates to seasons and drop the date column
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)
df= df.drop(columns=['Date'])

# Exercise 2. Define the feature and target dataframes
X = df.drop(columns='RainToday', axis=1)
y = df['RainToday']

# Exercise 3. How Balanced are these classes?
print(y.value_counts())