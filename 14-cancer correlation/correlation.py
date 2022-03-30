from statistics import linear_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error


def linear_error(X_train, X_valid, y_train, y_valid):
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds)


def nearest_neighbor_error(X_train, X_valid, y_train, y_valid, k):
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train.ravel())
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds)


if __name__ == "__main__":
    X_train = pd.read_csv('./14cancer.xtrain', delim_whitespace=True, header=None)
    X_train = X_train.iloc[:, :8]
    print(X_train.shape)

    # Y_train = pd.read_csv('./14cancer.ytrain', delim_whitespace=True, header=None)
    # print(Y_train.shape)
    # print(Y_train)
    # X_train.set_axis(Y_train,axis=1,inplace=True)

    X_train.corr()
    print(X_train.corr())
