import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def hyperparameter_tuning(X: df, y: np.ndarray) -> dict:
    """
    Perform hyperparameter tuning using Hyperopt for XGBoost

    Parameters:
    X: pd.DataFrame - feature matrix
    y: np.ndarray - target vector

    Returns:
    best_params: dict - selected parameters for XGBRegressor by Hyperopt

    """

    def objective(params):
        rmse_scores = []
        model = XGBRegressor(early_stopping_rounds=20)
        # Adjust parameters to model-specific format

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "eta": params["eta"],
            "max_depth": int(params["max_depth"]),
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "gamma": params["gamma"],
            "min_child_weight": int(params["min_child_weight"]),
            "lambda": params["lambda"],
            "alpha": params["alpha"],
        }

        X_train = X[~X.date_block_num.isin([33])]
        y_train = y.iloc[X_train.index]

        X_val = X[X["date_block_num"] == 33]
        y_val = y.iloc[X_val.index]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        y_pred = np.round(model.predict(X_val).clip(0, 20), 2)

        rmse = root_mean_squared_error(y_val, y_pred)

        # Average RMSE across folds
        return {"loss": rmse, "status": STATUS_OK}

    # Hyperparameter search space
    space = {
        "eta": hp.uniform("eta", 0.01, 0.3),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "gamma": hp.uniform("gamma", 0, 5),
        "min_child_weight": hp.quniform("min_child_weight", 50, 400, 25),
        "lambda": hp.uniform("lambda", 0, 1),
        "alpha": hp.uniform("alpha", 0, 1),
    }

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_child_weight"] = int(best_params["min_child_weight"])
    for key in best_params:
        if isinstance(best_params[key], float):
            best_params[key] = np.float16(best_params[key])

    print("Best parameters found:", best_params)

    return best_params
