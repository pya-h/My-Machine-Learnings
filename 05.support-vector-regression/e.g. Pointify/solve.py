import data_preprocessing
import matplotlib.pyplot as plt
import csv
import numpy as np

def create_points(start = -2, stop = 2, steps = 0.001):
    # Specify the CSV file name
    csv_file = 'points.csv'

    # Write array data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        i = start
        f = lambda x: x**3 + 1
        while i <= stop:
            writer.writerow([f(i), i])
            i += steps
    return csv_file


X, y_ex = data_preprocessing.read(filename=create_points())  # first column is equibalent to second
# like its already encoded

# Split the dataset; svr doesnt autoscale and not doing this will cause calculation problems
from sklearn.preprocessing import StandardScaler
x_scaler, y_scaler = StandardScaler(), StandardScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y_ex.reshape((len(y_ex), 1)))

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

plt.scatter(X, y, color='red'), plt.plot(X, regressor.predict(X), color='blue')
plt.title("SVR")
plt.legend(['Actual Values', 'Predicted Values'])
plt.xlabel('Position Level'), plt.ylabel('Salary')

plt.show()

# simple console app for point predicting:
while True:
    x = x_scaler.transform([[float(input('y = '))]])
    y_scaled = regressor.predict(x)
    y_svr = float(y_scaler.inverse_transform(y_scaled.reshape((len(y_scaled), 1))))

    print(f'\tx predicted by Support Vector Regresson Model:\t{y_svr}\n')
    if x in X:
        actual_value = float(y_ex[list(X).index(x)])
        print(f'\tActual x:\t{actual_value}')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        