# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
dataset = pd.read_csv('./multiple_linear_regression/50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# encode states with OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
#X = np.array(X)


# 80 / 20 split and no random

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state= 1)

print(len(X_train))
print(len(y_train))

print(len(X_test))
print(len(y_test))

# Fit multiple linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

# see how test would be predicted

y_predicted = regressor.predict(X_test)

#np.set_printoptions(precision=2)
for i in range(len(y_predicted)):    
    p = y_predicted[i]
    e = y_test[i]
    print('P: {:.1f} : E {:.1f}, {:.1f}%'.format(p,e,((p-e)/e)*100))

for item in zip(y_predicted,y_test):
    print(item)
    