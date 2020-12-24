#%%
# Data Preprocessing

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Dataset
data = pd.read_csv("")

#%%
# Features
X = data.iloc[:, :-1].values

#%%
# Check
print(X)

#%%
# Dependent Variable Vector
y = data.iloc[:, -1].values

#%%
# Check
print(y)

#%%
# Missing Data
from sklearn.impute import SimpleImputer

# Strategy: Mean, Median, Modus
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

#%%
# Check
print(X)

#%%
# One Hot Encoding for Categorical Data

#%%
# Encoding The Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#%%
# Check 
print(X)

#%%
# Encoding The Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#%%
# Check 
print(y)

#%%
# Splitting The Dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#%%
# Check 
print(X_train)

#%%
# Check 
print(X_test)

#%%
# Check 
print(y_train)

#%%
# Check 
print(y_test)

#%%
# Feature Scaling
# Standardization or Normalisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

#%%
# Check 
print(X_train)

#%%
# Check 
print(X_test)