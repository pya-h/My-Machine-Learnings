import data_preprocessing
from sklearn.model_selection import train_test_split

import numpy as np

X, y = data_preprocessing.read()

data_preprocessing.auto_impute(X)
X = data_preprocessing.hot_encode(X)

# prevent the dummy variable trap => remove one of them
X = X[:, 1:]

# split to test/train   
linear_regressor, y_predicted = data_preprocessing.linear_regression_prediction(X, y)

# Backward Elimination
    # current formula is: y = a1x1 + a2x2 + ...
# add ones to the start of the start of matrix for obtaining the optimal formula as:
    # (known as intercept)
    # y = x0 + a1x1 + a2x2


X_optimized = data_preprocessing.backward_eliminatation(X, y, SIGFICANT_LEVEL=0.05)

opt_linear_regressor, y_opt_predicted = data_preprocessing.linear_regression_prediction(X, y)
