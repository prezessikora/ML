# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
dataset = pd.read_csv('./data_preparation/Data.csv')

X = dataset[['Country','Age','Salary']].values
y = dataset['Purchased'].values

# Add missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

categorical_features = ['Country']

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = ct.fit_transform(X)
X = np.array(X)

# Use LabelEncoder to encode binary categorical data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

# 80 / 20 split and no random

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state= 1)

# Train set

print(X_train)
print(y_train)

# Test set

print(X_test)
print(y_test)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])