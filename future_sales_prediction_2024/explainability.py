import pandas as pd
from pandas.core.frame import DataFrame as df
import numpy as np
import shap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from pandas.core.frame import DataFrame as df


class Explainability:
    """
    Class initialization

    Parameters:
    - model: Trained model (e.g., XGBRegressor, LGBMRegressor, etc.)
    - X:np.ndarray - feature matrix
    - output_dir: Directory where results will be saved (default: "explainability_outputs")

    """

    def __init__(
        self, model, X: np.ndarray, output_dir: str = "explainability_outputs"
    ):

        self.model = model
        self.X = X
        self.explainer = shap.Explainer(model)
        self.shap_values = self.explainer(self.X)
        self.output_dir = output_dir

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def save_plot(self, plot_func, file_name: str):
        """
        Save a SHAP plot to a file, overwriting it if it exists

        Parameters:
        - plot_func: Function that generates the SHAP plot (e.g., shap.plots.bar)
        - file_name: Name of the file to save the plot

        """
        file_path = os.path.join(self.output_dir, file_name)
        # Create a new figure to avoid overlapping plots
        plt.figure()
        plot_func()
        plt.savefig(file_path, bbox_inches="tight")
        # Close the figure to free memory
        plt.close()
        print(f"Plot saved to: {file_path}")

    def explaine_instance(
        self, instance: df = None, file_name: str = "instance_explanation.png"
    ) -> shap.waterfall_plot:
        """
        Explain a single prediction using SHAP values

        Parameters:
        - instance: DataFrame containing a single row of data for which to generate explanation
                    If None, a random instance from X is used
        - file_name: Name of the file to save the plot (default: "instance_explanation.png")
        Returns:
        shap.waterfall_plot - display explanations for instance
        """
        if instance is None:
            instance = self.X.sample(1)

        shap_values_instance = self.explainer(instance)
        print("SHAP explanation for one instance")
        self.save_plot(lambda: shap.plots.waterfall(shap_values_instance[0]), file_name)

    def global_feature_importance(
        self, file_name: str = "global_feature_importance.png"
    ) -> shap.plots.bar:
        """
        Generate a SHAP summary plot showing global feature importance across the dataset

        Parameters:
        - file_name: Name of the file to save the plot (default: "global_feature_importance.png")

        Returns:
        shap.plots.bar
        """

        print("Global feature importance (SHAP values):")
        self.save_plot(lambda: shap.plots.bar(self.shap_values), file_name)

    def feature_dependence(
        self, feature_name: str, file_name: str = None
    ) -> shap.plots.scatter:
        """
        Generate a SHAP scatter plot for a given feature

        Parameters:
        - feature_name: Name of the feature to analyze for dependence
        - file_name: Name of the file to save the plot
                     If None, defaults to "{feature_name}_dependence.png"

        Returns:
        shap.dependence_plot
        """

        if file_name is None:
            file_name = f"{feature_name}_dependence.png"

        print(f"Generating SHAP dependence plot for {feature_name}:")
        self.save_plot(
            lambda: shap.plots.scatter(
                self.shap_values[:, feature_name], color=self.shap_values
            ),
            file_name,
        )


class ErrorAnalysis:

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model=XGBRegressor(),
        output_dir: str = "error_analysis_outputs",
    ):
        """
        Class initialization

        Parameters:
        X: np.ndarray - feature matrix
        y: np.ndarray - target matrix
        model: The trained model (default: XGBRegressor)
        output_dir: Directory where results will be saved (default: "error_analysis_outputs")
        """
        self.X = X
        self.y = y
        self.model = model
        self.X_val = None
        self.y_true = None
        self.y_pred = None
        self.error = None
        self.output_dir = output_dir

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def save_plot(self, plot_func, file_name: str):
        """
        Save a plot to a file, overwriting it if it exists

        Parameters:
        - plot_func: Function that generates the plot
        - file_name: Name of the file to save the plot
        """
        file_path = os.path.join(self.output_dir, file_name)
        plt.figure()
        plot_func()
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {file_path}")

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

    def model_drawbacks(self, file_name: str = "error_distribution.png"):
        """
        Model Performance by MAE and RMSE measurements

        Parameters:
        - file_name: Name of the file to save the error distribution plot

        """
        self.error = self.y_true - self.y_pred
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)

        print(f"Root mean squared error: {rmse}")
        print(f"Mean absolute error: {mae}")

        def plot_error_distribution():
            plt.hist(self.error, bins=50, color="skyblue", edgecolor="black")
            plt.title("Error Distribution")
            plt.xlabel("Errors")
            plt.ylabel("Frequency")

        self.save_plot(plot_error_distribution, file_name)

    def large_target_error(self, file_name: str = "large_target_error.png"):
        """
        Analyzes errors where the target values are large, checking for poor prediction performance

        Parameters:
        - file_name: Name of the file to save the error scatter plot

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
        def plot_large_target_error():
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

        self.save_plot(plot_large_target_error, file_name)

    def influence_on_error_rate(self, file_name: str = "influential_samples.png") -> df:
        """
        Identifies samples that have a significant influence on the model's error rate

        Parameters:
        - file_name: Name of the file to save the influential samples plot

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

        def plot_influential_samples():
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

        self.save_plot(plot_influential_samples, file_name)

        return influential_samples
