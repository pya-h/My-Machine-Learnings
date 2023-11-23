import data_preprocessing
import matplotlib.pyplot as plt
import numpy as np

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
poly_reg = PolynomialFeatures(degree=5)
# now transform the one column ind. variable matrix to a multy column matrix
# based on the raised terms of x
X_poly = poly_reg.fit_transform(X)
# then we create a linear regression model for new matrix [a0 a1x a2x^2 a3x^3 ...]
# y = a_0 + a_1.X + a_2.X^2 + a_3.X^3 + ... + a_(degree).X^degree
poly_model = data_preprocessing.get_linear_regressor(X_poly, y)

# plot & compare
plt.figure(figsize=(15, 8))
plot_inputs = ((slr_model.predict(X), "Linear Regression Model"),
               (poly_model.predict(X_poly), "Polynomial Regression Model"))

# create an X-axis for prediction plot, by creating a matrix stepping by 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plot_inputs = ((X, slr_model.predict(X), "Linear Regression Model"),
               (X, poly_model.predict(X_poly), "Polynomial Regression Model"),
               (X_grid, poly_model.predict(poly_reg.fit_transform(X_grid)), "Polynomial Regression Model (step=0.1)"))

for index, args in enumerate(plot_inputs, start=1):
    X_p, y_p, title = args
    plt.subplot(130 + index), plt.scatter(X, y, color='red'), plt.plot(X_p, y_p, color='blue')
    plt.title(title)
    plt.legend(['Actual Values', 'Predicted Values'])
    plt.xlabel('Position Level'), plt.ylabel('Salary')

plt.show()

# simple console app for point predicting:
while True:
    x = float(input('Enter the position level: '))
    y_lin = float(slr_model.predict([[x]]))
    y_pol = float(poly_model.predict(
        poly_reg.fit_transform([[x]])
    ))
    print(f'\tSalary predicted by Linear Regression Model:\t{y_lin}\n',
          f'\tSalary predicted by Polynomial Regression Model:\t{y_pol}')
    if x in X:
        actual_value = float(y[list(X).index(x)])
        print(f'\tActual Salary:\t{actual_value}')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        