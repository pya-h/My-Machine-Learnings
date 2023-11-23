import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# distinguish dependant and independant variables
# from the dataset structure its obvious that 'Purchased' is a dependent var (Y) and
# others are independent (X)
def read(filename: str='50_Startups.csv', X_start: int=0):
    # read the dataset
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, X_start:-1].values
    Y = dataset.iloc[:, -1].values
    return X, Y


def is_vector(x):
    return len(x.shape) == 1
    # return isinstance(x, (list, tuple, np.ndarray))
    # return hasattr(x, "__len__") and not np.isscalar(x[0])

# managing missing data


def custom_impute(X):
    missimputer = SimpleImputer(
        missing_values=np.nan, strategy='mean')  # , axis=0)
    # axis = 0 => means that the imputing operation will be done along a column (axis = 1 is for row)
    # fit the imputer to all the second and third column data
    missimputer = missimputer.fit(X[:, 1:3])
    # finally apply the imputer that is featured for
    X[:, 1:3] = missimputer.transform(X[:, 1:3])
    # putting the mean of the others items inside the missing value's column (2nd and 3rd)

# custom_impute(X)


# auto impute will automatically search for columns that contain null or nan items and impute them
def auto_impute(matrix):
    if not is_vector(matrix):
        cols = len(matrix[0, :])
        for i in range(cols):
            if np.any(pd.isna(matrix[:, i])) or np.any(pd.isnull(matrix[:, i])):
                missimputer = SimpleImputer(
                    missing_values=np.nan, strategy='mean')
                missimputer = missimputer.fit(matrix[:, i:i+1])
                matrix[:, i:i+1] = missimputer.transform(matrix[:, i:i+1])
    else:
        for i in range(len(matrix)):
            if np.any(pd.isna(matrix[:])) or np.any(pd.isnull(matrix[:])):
                missimputer = SimpleImputer(
                    missing_values=np.nan, strategy='mean')
                missimputer = missimputer.fit(matrix[:])
                matrix[:] = missimputer.transform(matrix[:])
    return matrix


def is_nan(x):
    for el in x:
        # if str(el).isnumeric() or np.isscalar(el):
        try:
            float(el)
            return False
        except:
            pass
    return True


# SIMPLE ENCODING
# this method uses a simple encoding method, that assigna specific number
# to each non numeric value, such as 0 => Germany, 1 => Spain, ...
def simple_encode(matrix, single_column=None):
    lblen = LabelEncoder()
    if single_column is not None:
        matrix[:, single_column] = lblen.fit_transform(
            matrix[:, single_column])
        return matrix

    if not is_vector(matrix):
        cols = len(matrix[0, :])
        for i in range(cols):
            if is_nan(matrix[:, i]):
                matrix[:, i] = lblen.fit_transform(matrix[:, i])
    else:
        if is_nan(matrix):
            matrix[:] = lblen.fit_transform(matrix[:])
    return matrix


# HOT ENCODING
# encode the columns that contain non-numeric values such as X[:, 0] and Y[:]
def hot_encode(matrix, single_column=None):

    def hot_encode_single(matrix, column):
        column_trans = ColumnTransformer(
            [("Whatever", OneHotEncoder(), [column])], remainder="passthrough")
        matrix = column_trans.fit_transform(matrix)
        return matrix

    if single_column is not None:
        #lblhoten = OneHotEncoder(categories_features=[single_column])
        #matrix = lblhoten.fit_transform(matrix).toarray()
        return hot_encode_single(matrix, single_column)

    if not is_vector(matrix):
        cols = len(matrix[0])
        i = 0
        while i < cols:
            if is_nan(matrix[:, i]):
                previous_length = len(matrix[0])
                matrix = hot_encode_single(matrix, i)
                cols = len(matrix[0])
                i += cols - previous_length
            i += 1

    else:
        if is_nan(matrix):
            matrix = hot_encode_single(matrix, 0)
    return matrix

def add_ones(matrix):
    return np.append(arr=np.ones((len(matrix), 1)).astype(int), values=matrix, axis=1)
    # arr and values parameters are used instead of eachother, so the ones become at the start of the matrix
    

def get_linear_regressor(X, y):
    return LinearRegression().fit(X, y)

    
def linear_regression_prediction(X, y, test_size: float=0.2, random_state: int=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor, regressor.predict(X_test)
    

def backward_eliminatation(X, y, SIGFICANT_LEVEL: float = 0.05):
    X = add_ones(X)  # REMEMBER: this doesnt change the actual X outside the function
    X_optimized = X[:, :]
    # start of Backward Elimination:
    while np.any(X_optimized):  # X_optimized is not empty
        X_optimized = np.array(X_optimized, dtype=float)  # this line fixes: ufunc 'isfinite' not supported for the input types, 
        OLS_regressor = sm.OLS(endog=y, exog=X_optimized).fit()
        print("X_opt=", X_optimized, " Summary: ", OLS_regressor.summary())
        maxp_index = np.argmax(OLS_regressor.pvalues)  # index of maximum p
        if maxp_index < SIGFICANT_LEVEL:  # end of the back-elimination algorythm
            break
        # if P > SL => remove predictor
        X_optimized = np.delete(X_optimized, maxp_index, axis=1)
    return X_optimized


if __name__ == '__main__':
    X, Y = read()
    auto_impute(X)  # didn't work for now :(
    # simple_encode(X)
    X = hot_encode(X)
    # since Y is a dependent var, machine learning algorithms know that Y is a category
    simple_encode(Y)
    # and there is no order between its values => LabelEncoder is good enough

    # now its time to split the data into training and test data
    # out machine learning project will earn on the training data and will check its understandings on the test data

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size=0.25, random_state=0)

    # now we transform the values into a common range so that all independent variables have the same effect
    # on the dependent variable

    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
