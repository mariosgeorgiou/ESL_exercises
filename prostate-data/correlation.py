import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    Data = pd.read_csv('./prostate.data', delim_whitespace=True)
    Data = Data[Data['train'] == 'T']
    Data = Data.drop(columns='train')
    print(Data.corr().round(5))

    # Data_norm = (Data-Data.mean())/Data.std()

    scaler = StandardScaler()
    Data_strd = scaler.fit_transform(Data)
    Data_strd = pd.DataFrame(Data_strd, columns=Data.columns)

    X_train = Data.iloc[:, :-1]
    y_train = Data.iloc[:, -1]

    # print(Data_norm - Data_strd)

    model = LinearRegression().fit(X_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    # print(model.)
