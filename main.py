import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the polynimial regression model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Visualising the linear regression model
# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg.predict(X), color = 'blue')
# plt.title("Truth or bluff (Linear Regression Model)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# Visualising the Polynomial Linear regression model with higher result
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, lin_reg_poly.predict(poly_reg.fit_transform(X_grid)), color='blue')
# plt.title("Truth or bluff (Linear Regression Model)")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.show()

# predicting new result with linear regression
lin_predict = lin_reg.predict([[6.5]])
print(lin_predict)

# Predictiing new result with Polynomial regression
poly_predict =lin_reg_poly.predict(poly_reg.fit_transform([[10]]))
print(poly_predict)