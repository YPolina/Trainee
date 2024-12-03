import pandas as pd
from pandas.core.frame import DataFrame as df
import numpy as np
import shap
from typing import Optional, Union
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.frame import DataFrame as df
import argparse
import os
import gcsfs
from future_sales_prediction_2024.data_handling import reduce_mem_usage


def loader(gcs_path: str) -> df:
    """
    Load data from a Google Cloud Storage path

    Parameters:
    - gcs_path: str - Google Cloud Storage path

    Returns:
    data: pd.DataFrames - data from .csv file
    """
    with fs.open(gcs_path) as f:
        return pd.read_csv(f)


class FeatureExtractor:
    def __init__(self, full_data: df, train: df):
        """
        Initialize with an existing DataFrame (full_data) for feature extraction

        Parameters:
        full_data: pd.DataFrame - Pre-existing full data containing required columns
        train: pd.DataFrame - Training data for aggregating revenue-based features
        """
        self.full_data = full_data
        self.train = train

    def history_features(self, agg: list, new_feature: str) -> df:
        """
        Adds a feature counting the number of unique months for which each combination in `agg` has sales data.

        Parameters:
        - agg: list - List of columns to group by (e.g., ['shop_id', 'item_id']).
        - new_feature: str - Name of the new feature to add.

        Returns:
        - pd.DataFrame - DataFrame with the additional feature based on historical sales counts.
        """
        group = (
            self.full_data[self.full_data.item_cnt_month > 0]
            .groupby(agg)["date_block_num"]
            .nunique()
            .rename(new_feature)
            .reset_index()
        )
        self.full_data = self.full_data.merge(group, on=agg, how="left")

    def feat_from_agg(self, df: df, agg: list, new_col: str, aggregation: list) -> df:
        """
        Aggregates features based on specified columns, aggregation functions, and adds the result as a new feature.

        Parameters:
        - agg: list - Columns to group by (e.g., ['shop_id', 'item_id']).
        - new_col: str - Name for the new aggregated feature.
        - aggregation: Dict[str, Union[str, List[str]]] - Aggregation functions to apply on the grouped data

        Returns:
        - pd.DataFrame - DataFrame with the new aggregated feature.
        """
        temp = (
            df[df.item_cnt_month > 0]
            if new_col == "first_sales_date_block"
            else df.copy()
        )
        temp = temp.groupby(agg).agg(aggregation)
        temp.columns = [new_col]
        temp.reset_index(inplace=True)
        self.full_data = pd.merge(self.full_data, temp, on=agg, how="left")

        if new_col == "first_sales_date_block":
            self.full_data.fillna(34, inplace=True)

    def lag_features(self, col: str, lags: list) -> df:
        """
        Adds lagged features to the DataFrame for specified columns over defined lag periods.

        Parameters:
        - col: str - Column to create lags for.
        - lags: list - List of lag periods to apply.

        Returns:
        - pd.DataFrame - DataFrame with the newly created lagged features.
        """
        temp = self.full_data[["date_block_num", "shop_id", "item_id", col]]
        for lag in lags:
            shifted = temp.copy()
            shifted.columns = [
                "date_block_num",
                "shop_id",
                "item_id",
                f"{col}_lag_{lag}",
            ]
            shifted["date_block_num"] += lag
            self.full_data = pd.merge(
                self.full_data,
                shifted,
                on=["date_block_num", "shop_id", "item_id"],
                how="left",
            )

    def new_items(self, agg: list, new_col: str) -> df:
        """
        Adds a feature tracking average monthly sales for items with specific historical conditions (e.g., item history of 1).

        Parameters:
        - agg: list - Columns to group by (e.g., ['shop_id', 'item_id']).
        - new_col: str - Name for the new column.

        Returns:
        - pd.DataFrame - DataFrame with the new column based on items' sales history.
        """

        temp = (
            self.full_data.query("item_history == 1")
            .groupby(agg)["item_cnt_month"]
            .mean()
            .reset_index()
            .rename(columns={"item_cnt_month": new_col})
        )
        self.full_data = self.full_data.merge(temp, on=agg, how="left")

    def add_revenue_features(self):
        """Add revenue-based features and lags

        Returns:
        - pd.DataFrame - DataFrame with revenue lags.
        """
        # Revenue-based aggregations
        revenue_agg_list = [
            (
                self.train,
                ["date_block_num", "item_category_id", "shop_id"],
                "sales_per_category_per_shop",
                {"revenue": "sum"},
            ),
            (
                self.train,
                ["date_block_num", "shop_id"],
                "sales_per_shop",
                {"revenue": "sum"},
            ),
            (
                self.train,
                ["date_block_num", "item_id"],
                "sales_per_item",
                {"revenue": "sum"},
            ),
        ]
        for df, agg, new_col, aggregation in revenue_agg_list:
            self.feat_from_agg(df, agg, new_col, aggregation)

        # Lag features for revenue aggregations
        revenue_lag_dict = {
            "sales_per_category_per_shop": [1],
            "sales_per_shop": [1],
            "sales_per_item": [1],
        }
        for feature, lags in revenue_lag_dict.items():
            self.lag_features(feature, lags)
            self.full_data.drop(columns=[feature], inplace=True)

    def add_item_price_features(self):
        """Add item price-related features, including delta revenue

        Returns:
        - pd.DataFrame - DataFrame with item_price and revenue lags.
        """
        # Average sales per shop for delta revenue
        self.feat_from_agg(
            self.train, ["shop_id"], "avg_sales_per_shop", {"revenue": "mean"}
        )
        self.full_data["avg_sales_per_shop"] = self.full_data[
            "avg_sales_per_shop"
        ].astype(np.float32)
        self.full_data["delta_revenue_lag_1"] = (
            self.full_data["sales_per_shop_lag_1"]
            - self.full_data["avg_sales_per_shop"]
        ) / self.full_data["avg_sales_per_shop"]
        self.full_data.drop(
            columns=["avg_sales_per_shop", "sales_per_shop_lag_1"], inplace=True
        )

        # Average item price features
        self.feat_from_agg(
            self.train, ["item_id"], "item_avg_item_price", {"item_price": "mean"}
        )
        self.full_data["item_avg_item_price"] = self.full_data[
            "item_avg_item_price"
        ].astype(np.float16)

        self.feat_from_agg(
            self.train,
            ["date_block_num", "item_id"],
            "date_item_avg_item_price",
            {"item_price": "mean"},
        )
        self.full_data["date_item_avg_item_price"] = self.full_data[
            "date_item_avg_item_price"
        ].astype(np.float16)

        # Lag for item price feature and delta price calculation
        self.lag_features("date_item_avg_item_price", [1])
        self.full_data["delta_price_lag_1"] = (
            self.full_data["date_item_avg_item_price_lag_1"]
            - self.full_data["item_avg_item_price"]
        ) / self.full_data["item_avg_item_price"]
        self.full_data.drop(
            columns=[
                "item_avg_item_price",
                "date_item_avg_item_price",
                "date_item_avg_item_price_lag_1",
            ],
            inplace=True,
        )

    def process(self):
        """Execute feature extraction on full_data

        Returns:
        - pd.DataFrame - full data with all features
        """
        # History features
        history = [
            ("shop_id", "shop_history"),
            ("item_id", "item_history"),
            ("minor_category_id", "minor_category_history"),
        ]
        for group, new_feature in history:
            self.history_features([group], new_feature)

        # Features from aggregations
        agg_list = [
            (
                self.full_data,
                ["date_block_num", "item_category_id"],
                "avg_item_cnt_per_cat",
                {"item_cnt_month": "mean"},
            ),
            (
                self.full_data,
                ["date_block_num", "city_id", "shop_id"],
                "avg_item_cnt_per_city_per_shop",
                {"item_cnt_month": "mean"},
            ),
            (
                self.full_data,
                ["date_block_num", "shop_id"],
                "avg_item_cnt_per_shop",
                {"item_cnt_month": "mean"},
            ),
            (
                self.full_data,
                ["date_block_num", "item_category_id", "shop_id"],
                "avg_item_cnt_per_cat_per_shop",
                {"item_cnt_month": "mean"},
            ),
            (
                self.full_data,
                ["date_block_num", "item_id"],
                "avg_item_cnt_per_item",
                {"item_cnt_month": "mean"},
            ),
            (
                self.full_data,
                ["date_block_num", "item_category_id", "shop_id"],
                "med_item_cnt_per_cat_per_shop",
                {"item_cnt_month": "median"},
            ),
            (
                self.full_data,
                ["date_block_num", "main_category_id"],
                "avg_item_cnt_per_main_cat",
                {"item_cnt_month": "mean"},
            ),
            (
                self.full_data,
                ["date_block_num", "minor_category_id"],
                "avg_item_cnt_per_minor_cat",
                {"item_cnt_month": "mean"},
            ),
            (
                self.full_data,
                ["item_id"],
                "first_sales_date_block",
                {"item_cnt_month": "min"},
            ),
        ]
        for df, agg, new_col, aggregation in agg_list:
            self.feat_from_agg(df, agg, new_col, aggregation)

        # Lagged features
        lag_dict = {
            "avg_item_cnt_per_cat": [1],
            "avg_item_cnt_per_shop": [1, 3, 6],
            "avg_item_cnt_per_item": [1, 3, 6],
            "avg_item_cnt_per_city_per_shop": [1],
            "avg_item_cnt_per_cat_per_shop": [1],
            "med_item_cnt_per_cat_per_shop": [1],
            "avg_item_cnt_per_main_cat": [1],
            "avg_item_cnt_per_minor_cat": [1],
            "item_cnt_month": [1, 2, 3, 6, 12],
        }

        for feature, lags in lag_dict.items():
            self.lag_features(feature, lags)
            if feature != "item_cnt_month":
                self.full_data.drop(columns=[feature], inplace=True)

        # Revenue and item price-related features
        self.add_revenue_features()
        self.add_item_price_features()

        # Last sale and time since last sale features
        self.full_data["last_sale"] = self.full_data.groupby(["shop_id", "item_id"])[
            "date_block_num"
        ].shift(1)
        self.full_data["months_from_last_sale"] = (
            self.full_data["date_block_num"] - self.full_data["last_sale"]
        )
        self.full_data["months_from_first_sale"] = self.full_data[
            "date_block_num"
        ] - self.full_data.groupby(["shop_id", "item_id"])["date_block_num"].transform(
            "min"
        )
        self.full_data["months_from_last_sale"].fillna(-1, inplace=True)
        self.full_data.drop("last_sale", axis=1, inplace=True)
        # Fill NaNs
        self.full_data.fillna(0, inplace=True)

        reduce_mem_usage(self.full_data)

        return self.full_data


class FeatureImportanceLayer:

    def __init__(self, X: df, y: df, output_dir: str = "feature_importance_results"):
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

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full_data", required=True, help="Path to full_data.csv in GCS"
    )
    parser.add_argument("--train", required=True, help="Path to train.csv in GCS")
    parser.add_argument(
        "--outdir", required=True, help="Path in GCS to save processed data"
    )
    args = parser.parse_args()

    fs = gcsfs.GCSFileSystem()

    # Load data
    full_data = loader(args.full_data)
    train = loader(args.train)

    # Run feature extraction
    extractor = FeatureExtractor(full_data=full_data, train=train)
    full_featured_data = extractor.process()

    with fs.open(f"{args.outdir}/full_featured_data.csv", "w") as f:
        full_featured_data.to_csv(f, index=False)

    print(f"Full featured data saved to {args.outdir}")
