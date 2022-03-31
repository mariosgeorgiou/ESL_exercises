import pandas as pd


if __name__ == "__main__":
    X_train = pd.read_csv('./prostate.data', delim_whitespace=True)
    X_train = X_train[X_train['train'] == 'T']
    X_train = X_train.drop(columns='train')
    print(X_train.corr().round(5))

    X_train_norm = (X_train-X_train.mean())/X_train.std()
    print(X_train_norm.corr().round(5))
