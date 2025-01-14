import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

### Import data from the csv files

X = pd.read_csv("ElectricityPrice-QRT/X_train.csv")
y = pd.read_csv("ElectricityPrice-QRT/y_train.csv")
X_final = pd.read_csv("ElectricityPrice-QRT/X_test_final.csv")
#print(X)

### Find columns with NaN values

NaN_columns = X_final.isna().sum()
#print(NaN_columns)

### Treat NaN values by replacing NaN with mean of column

for i in X.columns[X.isnull().any(axis=0)]:   
    X[i].fillna(X[i].mean(),inplace=True)

for i in X_final.columns[X_final.isnull().any(axis=0)]:   
    X_final[i].fillna(X_final[i].mean(),inplace=True)    

for i in y.columns[y.isnull().any(axis=0)]:   
    y[i].fillna(y[i].mean(),inplace=True)

### Vectorize country data

mappingDict = {'DE': 0, 'FR': 1}

X['COUNTRY'] = X['COUNTRY'].map(mappingDict)
X_final['COUNTRY'] = X_final['COUNTRY'].map(mappingDict)

### Cross validation and scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)

cols = X_train.columns

transformer = ColumnTransformer(transformers=[('num', StandardScaler(), slice(3, len(cols)))],
                                remainder='passthrough')

X_train = pd.DataFrame(transformer.fit_transform(X_train), columns=cols)

keep = X_train[['ID','DAY_ID', 'COUNTRY']]
X_train[['ID','DAY_ID', 'COUNTRY']] = X_train[['GAS_RET','COAL_RET','CARBON_RET']]
X_train[['GAS_RET','COAL_RET','CARBON_RET']] = keep

X_test = pd.DataFrame(transformer.transform(X_test), columns=cols)

keep = X_test[['ID','DAY_ID', 'COUNTRY']]
X_test[['ID','DAY_ID', 'COUNTRY']] = X_test[['GAS_RET','COAL_RET','CARBON_RET']]
X_test[['GAS_RET','COAL_RET','CARBON_RET']] = keep

X_final = pd.DataFrame(transformer.transform(X_final), columns=cols)

keep = X_final[['ID','DAY_ID', 'COUNTRY']]
X_final[['ID','DAY_ID', 'COUNTRY']] = X_final[['GAS_RET','COAL_RET','CARBON_RET']]
X_final[['GAS_RET','COAL_RET','CARBON_RET']] = keep

### Create model_v0

### Dimensionality reduction 

pca = PCA(n_components=9)   
  
# Keep only the first six principal components 
reg = LinearRegression() 
pipeline = Pipeline(steps=[('pca', pca), 
                           ('reg', reg)]) 
  
# Fit the pipeline to the data 
pipeline.fit(X_train, y_train) 
  
# Predict the labels for the data 
y_pred = pipeline.predict(X_test)

#model_v0 = LinearRegression().fit(X_train, y_train)
#model_v0 = Lasso(alpha = 0.01).fit(X_train, y_train)

#y_pred = model_v0.predict(X_test)

print('mean_squared_error : ', mean_squared_error(y_test, y_pred))
print('mean_absolute_error : ', mean_absolute_error(y_test, y_pred))

""" y_submit_pedro = model_v0.predict(X_final)

submission = pd.DataFrame(y_submit_pedro)

submission['ID'] = X_final['ID']
submission.set_index('ID')
submission.rename(columns = {1:'TARGET'}, inplace = True)
columns_titles = ["ID","TARGET"]
submission=submission.reindex(columns=columns_titles)
#print(submission)

#nan_cols = [i for i in X_test.columns if X_test[i].isnull().any()]
#print(nan_cols)

submission.to_csv('sublinreg.csv', encoding='utf-8', index=False) """