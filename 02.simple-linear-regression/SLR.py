
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data preprocessing
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=2/3, test_size=1/3)

# no need to feature scaling and n/a imputing and encoding, according to dataset
# Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(xTrain, yTrain)

# use regressor to make predictions on xTest
yPredicted = linear_regressor.predict(xTest)
yPredictedOnTrainies = linear_regressor.predict(xTrain)
'''
plt.scatter(xTrain, yTrain)
plt.plot(xTrain, yPredictedOnTrainies, color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')
plt.show()

plt.scatter(xTest, yTest)
plt.plot(xTrain, yPredictedOnTrainies, color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experiece')
plt.ylabel('Salary')
plt.show() '''
figure, axs = plt.subplots(2)
figure.suptitle('Salary vs Experience')
for index, points in enumerate([{'x': xTrain, 'y': yTrain, 'color': 'red'}, {'x': xTest, 'y': yTest, 'color': 'green'}]):
    axs[index].scatter(points['x'], points['y'], color=points['color'])
    axs[index].plot(xTrain, yPredictedOnTrainies)
    axs[index].set_xlabel('Years of Experiece')
    axs[index].set_ylabel('Salary')
    
figure.show()
figure.waitforbuttonpress()