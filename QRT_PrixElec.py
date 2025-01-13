import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.feature_extraction.text import CountVectorizer 

### Import data from the csv files

X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")
X_final = pd.read_csv("X_test_final.csv")
#print(X)

### Find columns with NaN values

NaN_columns = X.isna().sum()
#print(NaN_columns)

### Treat NaN values by replacing NaN with mean of column

for i in X.columns[X.isnull().any(axis=0)]:   
    X[i].fillna(X[i].mean(),inplace=True)

for i in y.columns[y.isnull().any(axis=0)]:   
    y[i].fillna(y[i].mean(),inplace=True)

### Vectorize country data

mappingDict = {'DE': 0, 'FR': 1}

X['COUNTRY'] = X['COUNTRY'].map(mappingDict)

### Cross validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

### Create model_v0

model_v0 = LinearRegression().fit(X_train, y_train)

y_pred = model_v0.predict(X_test)

print(
  'mean_squared_error : ', mean_squared_error(y_test, y_pred))
print(
  'mean_absolute_error : ', mean_absolute_error(y_test, y_pred))