import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./svr_model/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)

# Scale the variables because of much difference between feature X and dependent variable y

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print("X:")
print(X)
print("y:")
print(y)

# Train the model

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

r = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

print(r)

# Plot the model

import matplotlib.pyplot as plt    
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color = 'red')

plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)),color='blue')

plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('SVR')

plt.show()

