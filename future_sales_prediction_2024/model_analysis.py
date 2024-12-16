import pandas as pd
from pandas.core.frame import DataFrame as df
import numpy as np
import shap
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from typing import Optional, Union
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from future_sales_prediction_2024.data_loader import DataLoader

loader = DataLoader()


class FeatureImportanceLayer:

    def __init__(self, X: df, y: df, output_dir: str = loader.config['artifacts']['feature_importancy_layer']):
        """
        Initialization of model

        Parameters:
        X: pd.DataFrame - feature matrix
        y: pd.DataFrame - target vector
        output_dir: str - directory to save plots
        """
        self.output_dir = output_dir
        self.X = X
        self.y = y
        self.baseline_model = None
        self.baseline_importance = None
        self.final_model_importance = None


    def save_plot(self, fig: Figure, file_name: str) -> None:
        """
        Save the plot to the output directory

        Parameters:
        - fig: matplotlib.figure.Figure - plot to be saved
        - file_name: str - name of the file (e.g., "baseline_importance.png")
        """
        file_path = os.path.join(self.output_dir, file_name)
        fig.savefig(file_path, bbox_inches="tight")
        print(f"Plot saved to {file_path}")

    def fit_baseline_model(
        self, n_estimators: int = 30, random_state: int = 42
    ) -> None:
        """
        Fit Baseline RandomForestRegressor and calculate feature importances
        """

        print("Fitting Baseline Random Forest Regressor")
        self.baseline_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=2,
            n_jobs=-1,
            max_depth=15,
        )
        self.baseline_model.fit(self.X, self.y)
        self.baseline_importance = self.baseline_model.feature_importances_

        print("Baseline importances calculated")

    def fit_final_model(
        self, model=XGBRegressor, params: Optional[dict] = None, use_shap: bool = False
    ) -> None:
        """
        Fit a final model with specified hyperparameters and calculate feature importances

        Parameters:
        - model: Any ML model with .feature_importances_ or .coef_ attribute
        - params: Model hyperparameters
        - use_shap: Use SHAP values if the model doesn't provide native feature importance
        """
        model = model or XGBRegressor()
        print(f"Fitting {type(model).__name__}")
        model.set_params(**(params or {}))

        # Train, validation split
        if hasattr(model, "fit"):
            if isinstance(model, XGBRegressor):
                X_train = self.X[~self.X.date_block_num.isin([33])]
                y_train = self.y.iloc[X_train.index]

                X_val = self.X[self.X["date_block_num"] == 33]
                y_val = self.y.iloc[X_val.index]
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
            else:
                model.fit(self.X, self.y)
        else:
            raise ValueError("The provided model does not have a fit method.")

        self.final_model_importance = self._calculate_importances(model, use_shap)
        print(f"{type(model).__name__} model fitted and feature importances calculated")

    def _calculate_importances(self, model, use_shap: bool = False) -> np.ndarray:
        """
        Calculate feature importances for the given model

        Parameters:
        - model: Trained model
        - use_shap: Whether to use SHAP values if native feature importances aren't available

        Returns:
        - np.ndarray: Feature importance values
        """
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        # Use absolute value for linear models.
        elif hasattr(model, "coef_"):
            return np.abs(model.coef_)
        # Aggregate SHAP values
        elif use_shap:
            explainer = shap.Explainer(model, self.X)
            shap_values = explainer(self.X)
            return np.abs(shap_values.values).mean(axis=0)
        else:
            raise ValueError(
                "Model does not support feature importances or SHAP values"
            )

    def plot_feature_importances(
        self,
        importance_values: np.ndarray,
        top_n: int = 30,
        file_name: str = "feature_importance.png",
        title: str = "Feature Importances",
    ) -> Figure:
        """
        Plot feature importances.

        Parameters:
        - importance_values: np.ndarray - feature importance values
        - top_n: int - number of top features to plot
        - file_name: str - name of the file to save the plot
        - title: str - title of the plot

        Returns:
        - Figure: Matplotlib figure object
        """
        feature_importances = pd.Series(importance_values, index=self.X.columns)
        top_features = feature_importances.nlargest(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_features, y=top_features.index, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Feature")

        self.save_plot(fig, file_name)
        plt.close(fig)

        print(f'{file_name} is saved in {self.output_dir}')

    def plot_baseline_importance(
        self, top_n: int = 30, file_name: str = "baseline_importance.png"
    ) -> None:
        """Plot feature importances for the baseline model"""
        if self.baseline_importance is None:
            raise ValueError('Baseline model is not fitted. Run "fit_baseline_model"')
        self.plot_feature_importances(
            self.baseline_importance,
            top_n,
            file_name,
            title="Baseline Model Feature Importances",
        )

    def plot_final_importance(
        self, top_n: int = 30, file_name: str = "final_model_importance.png"
    ) -> None:
        """Plot feature importances for the final model"""
        if self.final_model_importance is None:
            raise ValueError('Final model is not fitted. Run "fit_final_model"')
        self.plot_feature_importances(
            self.final_model_importance,
            top_n,
            file_name,
            title="Final Model Feature Importances",
        )



class Explainability:
    """
    Class initialization

    Parameters:
    - model: Trained model (e.g., XGBRegressor, LGBMRegressor, etc.)
    - X:np.ndarray - feature matrix
    - output_dir: Directory where results will be saved (default: "explainability_outputs")

    """

    def __init__(
        self, model, X: np.ndarray, output_dir: str = loader.config['artifacts']['explainability_layer']
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
        output_dir: str = loader.config['artifacts']['feature_importancy_layer']
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
