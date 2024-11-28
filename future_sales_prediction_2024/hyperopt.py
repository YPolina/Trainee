import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pandas.core.frame import DataFrame as df
import os


def hyperparameter_tuning(
    X: df, y: np.ndarray, model_class, param_space: dict, eval_fn, max_evals: int = 50
) -> dict:
    """
    Perform hyperparameter tuning using Hyperopt for XGBoost

    Parameters:
    X: pd.DataFrame - feature matrix
    y: np.ndarray - target vector
    model_class: callable - model class (e.g., XGBRegressor, RandomForestRegressor, etc.)
    param_space: dict - Hyperopt search space for model hyperparameters
    eval_fn: callable - evaluation function that returns a loss metric
    max_evals: int - number of evaluations to perform (default=50)

    Returns:
    best_params: dict - selected parameters for XGBRegressor by Hyperopt

    """

    def objective(params):
        """
        Objective function for hyperparameter optimization

        """
        try:
            model = model_class(**params)
            loss = eval_fn(model, X, y)
        except Exception as e:
            # If the model fails, return a high loss
            print(f"Error: {e}")
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

    return best_params
