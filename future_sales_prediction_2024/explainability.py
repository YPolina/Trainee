import pandas as pd
from pandas.core.frame import DataFrame as df
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

class Explainability:
    """
    Class initialization

    Parameters:
    - model: Trained model (e.g., XGBRegressor, LGBMRegressor, etc.)
    - X:np.ndarray - feature matrix

    """

    def __init__(self, model, X: np.ndarray):

        self.model = model
        self.X = X
        self.explainer = shap.Explainer(model)
        self.shap_values = self.explainer(self.X)

    def explaine_instance(self, instance: df = None) -> shap.waterfall_plot:
        """
        Explain a single prediction using SHAP values

        Parameters:
        - instance: DataFrame containing a single row of data for which to generate explanation
                    If None, a random instance from X is used
        Returns:
        shap.waterfall_plot - display explanations for instance
        """
        if instance is None:
            instance = self.X.sample(1)

        shap_values_instance = self.explainer(instance)
        print("SHAP explanation for one instance")
        shap.plots.waterfall(shap_values_instance[0], max_display=35)

    def global_feature_importance(self) -> shap.plots.bar:
        """
        Generate a SHAP summary plot showing global feature importance across the dataset

        Returns:
        shap.plots.bar
        """

        print("Global feature importance (SHAP values):")
        shap.plots.bar(self.shap_values, max_display=35)

    def feature_dependence(self, feature_name: str) -> shap.plots.scatter:
        """
        Generate a SHAP scatter plot for a given feature

        Parameters:
        - feature_name: Name of the feature to analyze for dependence

        Returns:
        shap.dependence_plot
        """

        print(f"Generating SHAP dependence plot for {feature_name}:")
        shap.plots.scatter(self.shap_values[:, feature_name], color=self.shap_values)


class ErrorAnalysis:

    def __init__(
        self, X: np.ndarray, y: np.ndarray, model: XGBRegressor = XGBRegressor()
    ):
        """
        Class initialization

        Parameters:
        X: np.ndarray - feature matrix
        y: np.ndarray - target matrix
        model: The trained XGBRegressor model
        """
        self.X = X
        self.y = y
        self.model = model
        self.X_val = None
        self.y_true = None
        self.y_pred = None
        self.error = None

    def train_predict(self):
        """
        Train model and make predictions

        """

        X_train = self.X[~self.X.date_block_num.isin([33])]
        y_train = self.y.loc[X_train.index]

        self.X_val = self.X[self.X["date_block_num"] == 33]
        self.y_true = self.y.loc[self.X_val.index]
        self.model.fit(X_train, y_train)

        self.y_pred = self.model.predict(self.X_val).clip(0, 20)

    def model_drawbacks(self):
        """
        Model Performance by MAE and RMSE measurements
        """
        self.error = self.y_true - self.y_pred
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)

        print(f"Root mean squared error: {rmse}")
        print(f"Mean absolute error: {mae}")

        plt.figure(figsize=(10, 6))
        plt.hist(self.error, bins=50, color="skyblue", edgecolor="black")
        plt.title("Error Distribution")
        plt.xlabel("Errors")
        plt.ylabel("Frequence")
        plt.show()

    def large_target_error(self):
        """
        Analyzes errors where the target values are large, checking for poor prediction performance

        """
        # Large targets over 0.9 quantile
        threshold_1 = self.y_true.quantile(0.9)
        large_target_idx = self.y_true > threshold_1
        # Errors of large targets
        errors_for_large = self.error[large_target_idx]

        rmse_for_large = root_mean_squared_error(
            self.y_true[large_target_idx], self.y_pred[large_target_idx]
        )
        mae_for_large = mean_absolute_error(
            self.y_true[large_target_idx], self.y_pred[large_target_idx]
        )

        print(f"RMSE for large target values (>{threshold_1}): {rmse_for_large}")
        print(f"MAE for large target values (>{threshold_1}): {mae_for_large}")

        # Resulting plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.y_true[large_target_idx],
            errors_for_large,
            color="salmon",
            edgecolor="black",
        )
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("True Target Value")
        plt.ylabel("Prediction Error")
        plt.title(f"Prediction Error for Large Target Values (>{threshold_1})")
        plt.show()

    def influence_on_error_rate(self) -> df:
        """
        Identifies samples that have a significant influence on the model's error rate

        Returns:
        influential_samples: pd.DataFrame - samples with signinicant influence

        """
        # Threshold over 0.9 quantile
        error_threshold = self.error.quantile(0.9)
        influential_idx = np.abs(self.error) > error_threshold
        influential_samples = self.X_val.loc[influential_idx]
        influential_errors = self.error[influential_idx]

        print(f"Number of influential samples: {influential_samples.shape[0]}")
        print(
            f"Proportion of influential samples: {100 * influential_samples.shape[0] / len(self.error):.2f}%"
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.y_true[influential_idx],
            influential_errors,
            color="purple",
            edgecolor="black",
        )
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("True Target Value")
        plt.ylabel("Prediction Error")
        plt.title("Influential Samples Impacting Error Rate")
        plt.show()

        return influential_samples