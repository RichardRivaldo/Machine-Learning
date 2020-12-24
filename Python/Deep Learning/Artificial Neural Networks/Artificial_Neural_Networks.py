#%%
# Artificial Neural Network

#%%
# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#%%
# Data Preprocessing

#%%
# Dataset
dataset = pd.read_csv("")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#%%
# Check
print(X)
print(y)

#%%
# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#%%
# Check
print(X)

#%%
# Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
# Feature Scaling
# Compulsory in Deep Learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# Building the ANN

#%%
# Initializing the ANN
ann = tf.keras.models.Sequential()

#%%
# Layers

#%%
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#%%
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#%%
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


#%%
# Training the ANN

#%%
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#%%
# Training the ANN
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


#%%
# Making Predictions and Evaluations

#%%
# Single Data Prediction
"""
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000

So, should we say goodbye to that customer?
"""

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of 
square brackets. That's because the "predict" method always expects a 2D array as the format of its 
inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.

Important note 2: Notice also that the "France" country was not input as a string in the last column but 
as "1, 0, 0" in the first three columns. That's because of course the predict method expects the 
one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" 
was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the 
dummy variables are always created in the first columns.
"""

#%%
# Making Prediction on Test Set
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#%%
# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

#%%
Accuracy = accuracy_score(y_test, y_pred)
print(Accuracy)