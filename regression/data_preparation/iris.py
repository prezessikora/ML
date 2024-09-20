# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the Iris dataset

df = pd.read_csv('iris.csv');

# Separate features and target

# vars are: sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target

X = df[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].values
y = df['target'].values

# Split the dataset into an 80-20 training-test set

from sklearn.model_selection import train_test_split

# 80 / 20 split and no random

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state= 42)

# Apply feature scaling on the training and test sets

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Print the scaled training and test sets

# Train set

print(X_train)
print(y_train)

# Test set

print(X_test)
print(y_test)