import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Training the Multiple Linear Regression model on the Training set
def create_linear_regressor(X_train, y_train):
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

# Training the Polynomial Regression model on the Training set
def create_poly_regressor(X_train, y_train):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)
    return (regressor,poly_reg)

# Training the SVR Regression model on the Training set
def create_svr_regressor(X_train, y_train):
    # Feature scaling

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)      
    
    y_train = sc_y.fit_transform(y_train.reshape(len(y_train),1))
        
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train.ravel())

    return (regressor,sc_X,sc_y)

# Training the Decision Tree model on the Training set
def create_dtree_regressor(X_train, y_train):
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    return regressor

# Training the Random Forrest model on the Training set
def create_random_forrest_regressor(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    return regressor

## Predicting the Test set results

def predit_no_scaler(X_test, regressor):
    return regressor.predict(X_test)

def predit_with_scaler(X_test, regressor, sc_X, sc_y):
    #a = regressor.predict(sc_X.transform(X_test))
    
    a = regressor.predict(sc_X.transform(X_test)).reshape(len(X_test),1)
    
    return sc_y.inverse_transform(a)

def calc_r2(a,b):
    from sklearn.metrics import r2_score 
    R_square = r2_score(a, b) 
    return R_square



# 1 Import CSV dataset

dataset = pd.read_csv('selecting_model/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2 Splitting the dataset into the Training set and Test set
y = y.reshape(len(y),1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# linear_regressor = create_linear_regressor(X_train, y_train)
# (ply_regressor,poly_reg) = create_poly_regressor(X_train, y_train)
(svr_regressor,sc_X,sc_y) = create_svr_regressor(X_train, y_train)
# dtree_regressor = create_dtree_regressor(X_train, y_train)
# forrest_regressor = create_random_forrest_regressor(X_train, y_train)




# y_pred_linear = predit_no_scaler(X_test,linear_regressor)
# y_pred_ply = predit_no_scaler(poly_reg.transform(X_test),ply_regressor)
y_pred_svr = predit_with_scaler(X_test,svr_regressor,sc_X,sc_y)
# y_pred_dtree = predit_no_scaler(X_test,dtree_regressor)
# y_pred_forrest = predit_no_scaler(X_test,forrest_regressor)
np.set_printoptions(precision=2)



# print(np.concatenate( (y_pred_linear.reshape(len(y_pred_linear),1), y_test.reshape(len(y_test),1)), 1))
# print(f"=== R2 for linear model: {calc_r2(y_pred_linear,y_test)}")
# print(f"=== R2 for polynomial model: {calc_r2(y_pred_ply,y_test)}")
print(f"=== R2 for SVR model: {calc_r2(y_pred_svr,y_test)}")
# print(f"=== R2 for dtree model: {calc_r2(y_pred_dtree,y_test)}")
# print(f"=== R2 for forrest model: {calc_r2(y_pred_forrest,y_test)}")


