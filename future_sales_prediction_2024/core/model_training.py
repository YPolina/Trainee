import numpy as np
from typing import Union
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.frame import DataFrame as df


# Data validation using TSS
def tss_cv(
    df: df,
    n_splits,
    model: Union[LinearRegression, XGBRegressor, LGBMRegressor],
    true_pred_plot: bool = True,
):
    """
    Performs cross-validation for time series data using specified regression model and calculates RMSE.

    Parameters:
    - df: pd.DataFrame - DataFrame with features and target variable.
    - n_splits: int - Number of cross-validation splits.
    - model: Union[LinearRegression, XGBRegressor, LGBMRegressor] - Model to use for prediction.

    Returns:
    - Tuple[np.ndarray, np.ndarray, Union[LinearRegression, XGBRegressor, LGBMRegressor]] - True and predicted values, and trained model.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    n = 0
    rmse = []

    X_test = df[df.date_block_num == 34].drop("item_cnt_month", axis=1)

    X = df[df.date_block_num != 34].drop("item_cnt_month", axis=1)
    y = df[df.date_block_num != 34]["item_cnt_month"]

    print(f"{type(model).__name__}")
    model = model

    for train_idxs, val_idxs in tss.split(X):

        X_train, X_val = X.iloc[train_idxs], X.iloc[val_idxs]
        y_train, y_val = y.iloc[train_idxs], y.iloc[val_idxs]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val).clip(0, 20)
        rmse.append(root_mean_squared_error(y_pred, y_val))
        print(f"RMSE for split {n+1}: {rmse[n]:.3f}")
        n += 1
    mean_rmse = np.round(np.mean(rmse),3)
    print(f"Mean RMSE for all splits: {mean_rmse}")

    # Plots true versus predicted values to assess model performance visually
    if true_pred_plot:

        plt.figure(figsize=(10, 6))

        # Difference
        sns.scatterplot(x=y_val, y=y_pred, color="blue", alpha=0.5, s=30, edgecolor="k")
        # If prediction will be equal to our target
        plt.plot(
            [y_val.min(), y_val.max()],
            [y_val.min(), y_val.max()],
            color="red",
            linestyle="--",
        )

        plt.title("True vs Predicted Values")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.show()
    return mean_rmse


# Split begore model fitting (with eval_set)
def data_split(df: df) -> np.ndarray:
    """
    Splits data into training, validation, and test sets for model evaluation.

    Parameters:
    - df: pd.DataFrame - DataFrame containing features and target variable.
    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] - Training, validation, and test sets for features and target.
    """

    X_test = df[df["date_block_num"] == 34].drop(columns="item_cnt_month")
    X_test = X_test.reset_index()
    del X_test["index"]

    X_train = df[~df.date_block_num.isin([34])]
    y_train = X_train["item_cnt_month"]
    del X_train["item_cnt_month"]

    return X_train, y_train, X_test


# Training
def train_predict(
    X: df,
    y: np.ndarray,
    X_test: df,
    model_: Union[XGBRegressor, LGBMRegressor],
    model_params: dict = None,
) -> np.ndarray:
    """
    Train model and make predictions

    Parameters:
    - X: pd.DataFrame - Feature matrix for training
    - y: np.ndarray - target for training
    - X_test: pd.DataFrame - Feature matrix for prediction
    - model: Union[LinearRegression, XGBRegressor, LGBMRegressor] - Trained model to use for predictions
    - model_params: Dict[str, Any] - Model parameters to be set using set_params

    Returns:
    - y_pred: np.ndarray - prediction
    """
    model = model_
    model.set_params(**(model_params or {}))

    if isinstance(model, LinearRegression):
        model.fit(X, y)
    else:
        X_train = X[~X.date_block_num.isin([33])]
        y_train = y.iloc[X_train.index]

        X_val = X[X["date_block_num"] == 33]
        y_val = y.iloc[X_val.index]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Make predictions
    y_pred = np.round(model.predict(X_test), 2).clip(0, 20)

    return y_pred, model