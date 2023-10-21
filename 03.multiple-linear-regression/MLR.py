import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X, y = data_preprocessing.read()

data_preprocessing.auto_impute(X)
X = data_preprocessing.hot_encode(X)

# prevent the dummy variable trap => remove one of them
X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predicted = regressor.predict(X_test)