import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# constants
(DATASET_FILENAME, ) = ('Data.csv', )
# read the dataset
dataset = pd.read_csv(DATASET_FILENAME)

# distinguish dependant and independant variables
# from the dataset structure its obvious that 'Purchased' is a dependent var (Y) and
# others are independent (X)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


def is_vector(x):
    return len(x.shape) == 1
    # return isinstance(x, (list, tuple, np.ndarray))
    # return hasattr(x, "__len__") and not np.isscalar(x[0])

# managing missing data
from sklearn.impute import SimpleImputer

def custom_impute(X):
    missimputer = SimpleImputer(missing_values=np.nan, strategy='mean') #, axis=0)
    # axis = 0 => means that the imputing operation will be done along a column (axis = 1 is for row)
    missimputer = missimputer.fit(X[:, 1:3])  # fit the imputer to all the second and third column data
    X[:, 1:3] = missimputer.transform(X[:, 1:3])  # finally apply the imputer that is featured for
    # putting the mean of the others items inside the missing value's column (2nd and 3rd)

#custom_impute(X)


# auto impute will automatically search for columns that contain null or nan items and impute them
def auto_impute(matrix):
    if not is_vector(matrix):
        cols = len(matrix[0, :])
        for i in range(cols):
            if np.any(pd.isna(matrix[:, i])) or np.any(pd.isnull(matrix[:, i])):
                missimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                missimputer = missimputer.fit(matrix[:, i:i+1])
                matrix[:, i:i+1] = missimputer.transform(matrix[:, i:i+1])
    else:
        for i in range(len(matrix)):
            if np.any(pd.isna(matrix[:])) or np.any(pd.isnull(matrix[:])):
                missimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# this method uses a simple encoding method, that assigna specific number
# to each non numeric value, such as 0 => Germany, 1 => Spain, ...
def simple_encode(matrix, single_column = None):
    lblen = LabelEncoder()
    if single_column is not None:
        matrix[:, single_column] = lblen.fit_transform(matrix[:, single_column])
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
from sklearn.compose import ColumnTransformer
# encode the columns that contain non-numeric values such as X[:, 0] and Y[:]
def hot_encode(matrix, single_column=None):
    
    def hot_encode_single(matrix, column):
        column_trans = ColumnTransformer([("Whatever", OneHotEncoder(), [column])], remainder="passthrough")
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


auto_impute(X)  # didnt work for now :(
# simple_encode(X)
X = hot_encode(X)
simple_encode(Y) # since Y is a dependant var, machine learning algorythms know that Y is a category
# and there is no order bettween its values => LabelEncoder is good enough

# now its time to split the data into trainging and test data
# out machine learning project will earn on the training data and will check its understandings on the test data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

# now we transform the values into a common range so that all independent variables have the same effect
# on the dependant variable 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)
