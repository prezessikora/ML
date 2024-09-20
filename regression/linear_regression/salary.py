# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('./regression/linear_regression/Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# 80 / 20 split and no random

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state= 1)

# Fit linear regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Plot train data and fitted model (line)
import matplotlib.pyplot as plt

def plotDataAndFit(Xv,yv):
    plt.scatter(Xv,yv, color = 'red')
    plt.plot(Xv,regressor.predict(Xv), color='blue')

    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.title('Experience vs Salary')
    plt.show()

plotDataAndFit(X_train,y_train)
plotDataAndFit(X_test,y_test)
