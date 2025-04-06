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

# Exercise 4. What can you conclude from these counts?
# it rains 1791 days out of a total of 7557 days leaving 5766 days with no rain. You would have about 76.3% accuaracy if you just assumed it won't rain everday. No there is a pretty large class imbalance and is therefore not a balanced dataset. We need to do a test/train split

# Exercise 5. Split data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Exercise 6. Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()  
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Exercise 7. Combine the transformers into a single preprocessing column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

## Exercise 8. Create a pipeline by combining the preprocessing with a Random Forest classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define a parameter grid to use in a cross validation grid search model optimizer
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True)

## Exercise 9. Instantiate and fit GridSearchCV to the pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)  
grid_search.fit(X_train, y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Exercise 10. Display your model's estimated score
test_score = grid_search.score(X_test, y_test)  
print("Test set score: {:.2f}".format(test_score))

## Exercise 11. Get the model predictions from the grid search estimator on the unseen data
y_pred = grid_search.predict(X_test)

## Exercise 12. Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Exercise 13. Plot the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()