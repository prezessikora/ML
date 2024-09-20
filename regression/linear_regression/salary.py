# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('./linear_regression/Salary_Data.csv')

#X = dataset['YearsExperience'].values
#y = dataset['Salary'].values


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# 80 / 20 split and no random

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state= 1)

# Fit linear regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test);

import matplotlib.pyplot as plt

# Plot train data and fitted model (line)

plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color='blue')

plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()

# Plot train data and fitted model (line)

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color='blue')

plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()