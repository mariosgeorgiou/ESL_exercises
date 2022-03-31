import pandas as pd


if __name__ == "__main__":
    X_train = pd.read_csv('./prostate.data', delim_whitespace=True)
    X_train = X_train[X_train['train'] == 'T']
    print(X_train.corr())
