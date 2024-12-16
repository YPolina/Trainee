import yaml
import json
import joblib
import os
import numpy as np
from typing import Union
from pandas.core.frame import DataFrame as df
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import seaborn as sns



class Trainer:
    def __init__(self, n_splits=3, config_path = 'config.yaml'):
        """
        Initialize the Trainer class
        
        Parameters:
        - n_splits: Number of Time Series CV splits
        - config_path: str - Path to the YAML configuration file.

        """
        self.n_splits = n_splits
        self.model = None
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

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


    def train_predict(self, X_train, y_train, X_test, model_name, model_params=None, save_model: bool = True):
        """
        Train the model and make predictions

        Parameters:
        - X_train: pd.DataFrame - Feature matrix for training
        - y_train: pd.Series - Target variable for training
        - X_test: pd.DataFrame - Feature matrix for prediction
        - model_params: dict - Model parameters to set
        - save_model: bool - Whether to save model or not

        Returns:
        - y_pred: np.ndarray - Predictions for the test set
        - model: The trained model
        """
        try:
            model_class = self._get_model_class(model_name)
            self.model = model_class(**model_params)
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

        if model_name == 'LinearRegression':
            pipeline = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("regressor", self.model)
            ])

            pipeline.fit(X_train_final, y_train_final)
            y_pred_val = np.round(pipeline.predict(X_val), 2).clip(0, 20)
            rmse_score = root_mean_squared_error(y_val, y_pred_val)
            self.model = pipeline
        elif hasattr(self.model, "eval_set"):

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

        if save_model:  

            # Save the trained model
            save_path = os.path.join(self.config['artifacts']['models'], 'trained_model.pkl')
            joblib.dump(self.model, save_path)

            print(f"trained_model.pkl saved in {save_path}")
        print(X_test)

        # Make predictions
        y_pred = np.round(self.model.predict(X_test), 2).clip(0, 20)

        return y_pred, self.model, rmse_score
        
class HyperparameterTuner:
    """
    A class to handle hyperparameter tuning using Hyperopt or grid search based on user preference
    """

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the tuner with model configurations.

        Parameters:
        - config_path: str - Path to the YAML configuration file.
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def _build_search_space(self, model_name):
        """
        Build the search space for the specified model using the config file

        Parameters:
        - model_name: str - The name of the model (e.g., "XGBRegressor")

        Returns:
        - param_space: dict - The parameter search space for the model
        """
        model_config = self.config["models"].get(model_name)
        param_space = {}
        for param, details in model_config["param_space"].items():
            if details["type"] == "uniform":
                param_space[param] = hp.uniform(param, details["low"], details["high"])
            elif details["type"] == "choice":
                param_space[param] = hp.choice(param, details["options"])
            elif details["type"] == "randint":
                param_space[param] = hp.randint(param, details["low"], details["high"])
            elif details["type"] == "fixed":
                param_space[param] = details["value"]
        return param_space

    def tune(self, X: df, y: np.ndarray, model_name: str = 'XGBRegressor', custom_params: dict = None, max_evals: int = 50):
        """
        Perform hyperparameter tuning.

        Parameters:
        - X: pd.DataFrame - Feature matrix
        - y: np.ndarray - Target vector
        - model_name: str - Name of the model to tune
        - custom_params: dict - Custom parameter space (overrides default if provided)
        - max_evals: int - Number of evaluations for Hyperopt

        Returns:
        - best_params: dict - Best hyperparameters found
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Model '{model_name}' is not supported. Check the config file")

        # Use custom params if provided, otherwise load from config
        param_space = custom_params if custom_params else self._build_search_space(model_name)

        def objective(params):
            """
            Objective function for hyperparameter tuning.
            """
            try:
                model_class = self._get_model_class(model_name)
                model = model_class(**params)
                loss = self._eval_fn(model, X, y)
            except Exception as e:
                print(f"Error during evaluation: {e}")
                return {"loss": float("inf"), "status": STATUS_OK}
            return {"loss": loss, "status": STATUS_OK}

        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42),
        )


        def convert_types(value):
            if isinstance(value, (np.int64, np.int32)):
                return np.int16(value).item()
            elif isinstance(value, (np.float64, np.float32)):
                return np.float16(value).item()
            else:
                return value

        best_params_converted = {key: convert_types(value) for key, value in best_params.items()}

        print("Best parameters are found:", best_params_converted)

        save_path = os.path.join(self.config['artifacts']['params'], 'best_params.json')
        # Save to file
        with open(save_path, "w") as f:
            json.dump(best_params_converted, f)

        print(f'Best params are saved in {save_path}')

        return best_params_converted, model_name

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

    def _eval_fn(self, model, X, y):

        """
        Placeholder for model evaluation logic
        Replace this with cross-validation or hold-out validation logic

        Parameters:
        - model: callable - The model to evaluate
        - X: pd.DataFrame - Feature matrix
        - y: np.ndarray - Target vector

        Returns:
        - loss: float - Loss metric (e.g., RMSE)
        """

        X_train = X[~X.date_block_num.isin([33])]
        y_train = y.iloc[X_train.index]

        X_val = X[X["date_block_num"] == 33]
        y_val = y.iloc[X_val.index]
        model.fit(X_train, y_train)
        y_pred = np.round(model.predict(X_val).clip(0, 20), 2)

        rmse = root_mean_squared_error(y_val, y_pred)
        return rmse

