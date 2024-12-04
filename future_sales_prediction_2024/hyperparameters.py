import yaml
import json
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from pandas.core.frame import DataFrame as df


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

    def tune(self, X: df, y: np.ndarray, model_name: str, custom_params: dict = None, max_evals: int = 50):
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

        print("Best parameters found:", best_params)

        # Create output directory if it doesn't exist
        os.makedirs('./attributes', exist_ok=True)

        # Save to file
        with open("./attributes/best_params.json", "w") as f:
            json.dump(best_params, f)

        print('Best params are saved in ./attributes/best_params.json')

        return best_params, model_name

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

    def _eval_fn(model, X, y):

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
