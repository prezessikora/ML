# Importing the necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv('./polynomial_regression/Position_Salaries.csv')

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Fit multiple linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,y)

# see how test would be predicted

y_predicted = regressor.predict(X)

#

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly,y)
y2_predicted = regressor2.predict(X_poly)

print("LINEAR")
for i in range(len(y_predicted)):    
    p = y_predicted[i]
    e = y[i]
    print('P: {:.1f} : E {:.1f}, {:.1f}%'.format(p,e,((p-e)/e)*100))

print("POLY LINEAR")

for i in range(len(y2_predicted)):    
    p = y2_predicted[i]
    e = y[i]
    print('P: {:.1f} : E {:.1f}, {:.1f}%'.format(p,e,((p-e)/e)*100))


#for item in zip(y_predicted,y):
 #   print(item)
    
import matplotlib.pyplot as plt    
plt.scatter(X,y,color = 'red')
plt.plot(X,y_predicted,color='blue')

plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Linear Regression Fit')


plt.show()


# Poly linear model plot

import matplotlib.pyplot as plt    
plt.scatter(X,y,color = 'red')
plt.plot(X,y2_predicted,color='blue')

plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Poly Linear Regression Fit')

plt.show()

print(X_poly)