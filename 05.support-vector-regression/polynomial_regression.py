import data_preprocessing
import matplotlib.pyplot as plt
import numpy as np

X, y_ex = data_preprocessing.read(filename='Position_Salaries.csv', X_start=1)  # first column is equibalent to second
# like its already encoded

# Split the dataset; svr doesnt autoscale and not doing this will cause calculation problems
from sklearn.preprocessing import StandardScaler
x_scaler, y_scaler = StandardScaler(), StandardScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y_ex.reshape((len(y_ex), 1)))

# dont split the data here, because the number of observations is limited here
# and in polynomial regression we need as much as data we have, to improve model accuracy


# svr regression:
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# plot & compare
plt.figure(figsize=(15, 8))
# create an X-axis for prediction plot, by creating a matrix stepping by 0.1
# this will enhance the plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plot_inputs = ((X, regressor.predict(X), "SVR"),
               (X_grid, regressor.predict(X_grid), "SVR (more accurate)"))

for index, args in enumerate(plot_inputs, start=1):
    X_p, y_p, title = args
    plt.subplot(130 + index), plt.scatter(X, y, color='red'), plt.plot(X_p, y_p, color='blue')
    plt.title(title)
    plt.legend(['Actual Values', 'Predicted Values'])
    plt.xlabel('Position Level'), plt.ylabel('Salary')

plt.show()

# simple console app for point predicting:
while True:
    x = x_scaler.transform([[float(input('Enter the position level: '))]])
    y_scaled = regressor.predict(x)
    y_svr = float(y_scaler.inverse_transform(y_scaled.reshape((len(y_scaled), 1))))

    print(f'\tSalary predicted by Support Vector Regresson Model:\t{y_svr}\n')
    if x in X:
        actual_value = float(y_ex[list(X).index(x)])
        print(f'\tActual Salary:\t{actual_value}')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        