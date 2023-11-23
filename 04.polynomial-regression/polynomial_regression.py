import data_preprocessing

X, y = data_preprocessing.read(filename='Position_Salaries.csv', X_start=1)  # first column is equibalent to second
# like its already encoded

# There is no need to scaling, because there's just one ind. variable as is supposed in polynomial regression

# We dont split the data here, because the number of observations is limited here
# and in polynomial regression we need as much as data we have, to improve model accuracy

# we create two models:
# first a simle linear regression model
slr_model = data_preprocessing.get_linear_regressor(X, y)

# 2nd: a polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
# now transform the one column ind. variable matrix to a multy column matrix
# based on the raised terms of x
X_poly = poly_reg.fit_transform(X)
# then we create a linear regression model for new matrix [a0 a1x a2x^2 a3x^3 ...]
# y = a0 + a1X + a2X^2 + a3X^3 + ...
poly_model = data_preprocessing.get_linear_regressor(X_poly, y)

