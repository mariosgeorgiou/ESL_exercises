from statistics import linear_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def linear_error(X_train, X_valid, y_train, y_valid):
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds)

def nearest_neighbor_error(X_train, X_valid, y_train, y_valid, k):
    model = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds)


if __name__ == "__main__":
    twos = pd.read_csv('zipcode/train.2').values
    threes = pd.read_csv('zipcode/train.3').values
    X_train = np.concatenate((twos,threes),axis=0)
    Y_two = np.full((twos.shape[0], 1),2.)
    Y_three = np.full((threes.shape[0], 1),3.)
    y_train = np.concatenate((Y_two, Y_three), axis=0)

    test = pd.read_csv('zipcode/zip.test', delim_whitespace=True, header=None)
    test = test[test.iloc[:, 0].isin({2,3})]
    X_valid = test.iloc[:, 1:].values
    y_valid = test.iloc[:, :1].values

    print(linear_error(X_train,X_train,y_train,y_train))
    print(linear_error(X_train,X_valid,y_train,y_valid))

    for i in {1,3,5,7,15}:
        print(nearest_neighbor_error(X_train,X_train,y_train,y_train,i))
        print(nearest_neighbor_error(X_train,X_valid,y_train,y_valid,i))