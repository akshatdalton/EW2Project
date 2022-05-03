import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("./Group20A.csv")

s1 = set(list(data["Process1"]))
s2 = set(list(data["Process2"]))
s3 = set(list(data["Process3"]))

s = s1.union(s2, s3)

replace_from_dict = {}
for ind, val in enumerate(s):
    replace_from_dict[val] = ind + 1

data = data.replace(replace_from_dict)

# Droping off the Parameter column as it is unique for each of the combinations.
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1:]


def get_rmse_and_variance_score(X, y, order):
    # include_bias True means keep constant term in the polynomial eqn as zero.
    poly = PolynomialFeatures(degree=order, include_bias=False)
    poly_features = poly.fit_transform(X)

    # formula: shape = 1 + (n_features * order)
    print("order = ", order, " poly_features.shape = ", poly_features.shape)

    # Split the data into 40% training and 60% testing
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.4, random_state=1)

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_train, y_train)

    poly_reg_y_predicted = poly_reg_model.predict(X_test)
    poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
    
    return poly_reg_rmse, poly_reg_model.score(X_test, y_test)


rmse, var_score = get_rmse_and_variance_score(X, y, 6)
print("RMSE = ", rmse)
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(var_score))


order_list = np.arange(1, 6)
rmse_list = []
variance_score_list = []

for order in order_list:
    rmse, var_score = get_rmse_and_variance_score(X, y, order)
    rmse_list.append(rmse)
    variance_score_list.append(var_score)


order = 5
# include_bias True means keep constant term in the polynomial eqn as zero.
poly = PolynomialFeatures(degree=order, include_bias=False)
poly_features = poly.fit_transform(X)

# formula: shape = 1 + (n_features * order)
print("order = ", order, " poly_features.shape = ", poly_features.shape)

# Split the data into 40% training and 60% testing
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.4, random_state=1)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)

poly_reg_y_predicted = poly_reg_model.predict(X_test)
poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))

print(replace_from_dict)

# Predict when Process1 = ss_lp_bjt, Process2 = tt_lp_rvt12, Process3 = tt_lp_io25, Temperature = 10, V-supply = 1.5
poly_features = poly.fit_transform(np.array([7, 1, 3, 10, 1.5]).reshape(1, -1))
# This gives vref
print(poly_reg_model.predict(poly_features))
