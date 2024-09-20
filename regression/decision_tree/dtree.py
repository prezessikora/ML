import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./svr_model/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(len(X))
print(len(y))

from sklearn import tree
clf = tree.DecisionTreeRegressor(random_state=0)
clf = clf.fit(X, y)

r = clf.predict([[8.5]])
print(r)

tree.plot_tree(clf)

plt.show()

import matplotlib.pyplot as plt    
plt.scatter(X,y,color = 'red')

plt.plot(X,clf.predict(X),color='blue')

plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('TREE')


plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,clf.predict(X_grid),color='blue')
plt.show()