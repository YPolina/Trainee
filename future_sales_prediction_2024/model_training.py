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
import joblib


class Trainer:
    def __init__(self, n_splits=3):
        """
        Initialize the Trainer class
        
        Parameters:
        - model: The regression model (e.g., LinearRegression, XGBRegressor, etc.)
        - n_splits: Number of Time Series CV splits

        """
        self.n_splits = n_splits

    def _get_model_class(self, model_name):
        """
        Retrieve the model class based on the name

        Parameters:
        - model_name: str - Name of the model (e.g., "XGBRegressor")

        Returns:
        - model_class: callable - The corresponding model class
        """
        model_classes = {
            "LightGBM": LGBMRegressor,
            "XGBRegressor": XGBRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
        }
        return model_classes.get(model_name)

    def split_data(self, df):
        """
        Split the data into training, validation, and test sets
        
        Returns:
        - X_train: pd.DataFrame - Training feature matrix
        - y_train: pd.Series - Training target variable
        - X_test: pd.DataFrame - Test feature matrix
        """
        X_test = df[df["date_block_num"] == 34].drop(columns="item_cnt_month")
        X_test = X_test.reset_index(drop=True)

        X_train = df[~df.date_block_num.isin([34])]
        y_train = X_train.pop("item_cnt_month")

        return X_train, y_train, X_test


    def train_predict(self, X_train, y_train, X_test, model_name, model_params=None):
        """
        Train the model and make predictions

        Parameters:
        - X_train: pd.DataFrame - Feature matrix for training
        - y_train: pd.Series - Target variable for training
        - X_test: pd.DataFrame - Feature matrix for prediction
        - model_params: dict - Model parameters to set

        Returns:
        - y_pred: np.ndarray - Predictions for the test set
        - model: The trained model
        """
        try:
            model_class = self._get_model_class(model_name)
            model = model_class(**model_params)
        except: 
            raise ValueError(f"Model '{model_name}' is not supported")

        best_metric = None
        best_iteration = None

        # Train/validation split
        train_mask = ~X_train.date_block_num.isin([33])
        X_train_final = X_train[train_mask]
        y_train_final = y_train.iloc[X_train_final.index]
        X_val = X_train[~train_mask]
        y_val = y_train.iloc[X_val.index]

        if hasattr(self.model, "eval_set"):

            self.model.fit(
                X_train_final,
                y_train_final,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                early_stopping_rounds=50,
            )

            # Extract the best metric
            if hasattr(self.model, "best_score"):
                rmse_score = self.model.best_score

        else:
            self.model.fit(X_train_final, y_train_final)
            y_pred_val = np.round(self.model.predict(X_val), 2).clip(0, 20)
            rmse_score = root_mean_squared_error(y_pred_val, y_val)

        # Create output directory if it doesn't exist
        os.makedirs('attributes/models', exist_ok=True)

        # Save the trained model
        joblib.dump(self.model, "./attributes/models/trained_model.pkl")

        print(f"trained_model.pkl saved in ./attributes/models")

        # Make predictions
        y_pred = np.round(self.model.predict(X_test), 2).clip(0, 20)

        return y_pred, self.model, rmse_score
