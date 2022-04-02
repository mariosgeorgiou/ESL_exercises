import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    Data = pd.read_csv('./prostate.data', delim_whitespace=True)
    Data_train = Data[Data['train'] == 'T']
    Data_train = Data_train.drop(columns='train')
    Data_test = Data[Data['train'] == 'F']
    Data_test = Data_test.drop(columns='train')
    print('Correlation Matrix')
    print(Data.corr().round(5))

    scaler = StandardScaler()
    Data_strd = scaler.fit_transform(Data_train)
    Data_strd = pd.DataFrame(Data_strd, columns=Data_train.columns)

    X_train = Data_strd.iloc[:, :-1]
    y_train = Data_strd.iloc[:, -1]

    X_train = sm.add_constant(X_train)
    model2 = sm.OLS(y_train, X_train)
    results = model2.fit()
    print(results.summary())

    hypothesis = '(age = 0), (lcp = 0), (gleason = 0), (pgg45 = 0)'
    print(results.f_test(hypothesis))


    Data_strd = scaler.fit_transform(Data_test)
    Data_strd = pd.DataFrame(Data_strd, columns=Data_test.columns)

    X_test = Data_strd.iloc[:, :-1]
    y_test = Data_strd.iloc[:, -1]

    X_test = sm.add_constant(X_test)
    y_pred = results.predict(X_test)
    print(mean_squared_error(y_pred, y_test))

