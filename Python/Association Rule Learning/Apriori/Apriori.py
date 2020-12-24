#%%
# Apriori

#%%
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Data Preprocessing
# Should be string to be processed
dataset = pd.read_csv("", header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#%%
# Apriori Model
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#%%
# Unprocessed Rules from Apriori
results = list(rules)
print(results)

#%%
# Make a Pandas DataFrame to contain the rules
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#%%
# Non-Sorted Rules
print(resultsinDataFrame)

#%%
# Sorted Rules with Descending Lift Values
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')