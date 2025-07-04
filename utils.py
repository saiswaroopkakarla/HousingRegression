import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def train_models(df):
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}

    return results



def tune_and_train_models(df):
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_params = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {
                "fit_intercept": [True, False],
                "positive": [True, False],
                "copy_X": [True, False]  
            }
        },
        "Ridge": {
            "model": Ridge(),
            "params": {
                "alpha": [0.1, 1.0, 10.0],
                "fit_intercept": [True, False],
                "solver": ['auto', 'svd', 'cholesky']
            }
        },
        "Lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [0.01, 0.1, 1.0],
                "selection": ['cyclic', 'random'],
                "max_iter": [1000, 5000]
            }
        }
    }

    results = {}

    for name, mp in models_params.items():
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            "Best Params": grid.best_params_,
            "MSE": mse,
            "R2": r2
        }

    return results

