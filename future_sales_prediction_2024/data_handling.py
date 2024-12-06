import pandas as pd
import gcsfs
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import product
from pandas.core.frame import DataFrame as df
import argparse
import yaml
from pathlib import Path
from future_sales_prediction_2024.data_loader import DataLoader


# Reducing memory usage
class MemoryReducer:
    """Reduces memory usage of pandas DataFrames."""

    @staticmethod
    def reduce(df: df, verbose: bool = True) -> None:
        """
        Reduces memory usage of a DataFrame by downcasting numeric columns

        Parameters:
        - df: pd.DataFrame - The DataFrame to optimize
        - verbose: bool - Whether to print memory usage details

        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = df.memory_usage().sum() / 1024**2

        for col in df.columns:
            col_type = df[col].dtypes

            if col_type in numerics:
                c_min, c_max = df[col].min(), df[col].max()

                # Optimize integer types
                if str(col_type)[:3] == "int":

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)

                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)

                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)

                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                # Optimize float types
                else:

                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)

                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)

                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2

        if verbose:
            print(
                f"Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
            )


# Creating df with full range of data
class DataPreparer:
    """Handles data preparation and feature engineering"""

    def __init__(self, memory_reducer: MemoryReducer):
        self.memory_reducer = memory_reducer

    @staticmethod
    def full_data_creation(df: df, agg_group: list, periods: int) -> df:
        """
        Generates a DataFrame with the full range of specified item and shop combinations for each period

        Parameters:
        - df: pd.DataFrame - Input DataFrame with existing data
        - agg_group: list - Columns to aggregate
        - periods: int - Number of periods to include in the generated DataFrame

        Returns:
        - full_data: pd.DataFrame - DataFrame containing all combinations of items, shops, and periods
        """
        full_data = []

        for i in range(periods):
            sales = df[df.date_block_num == i]
            full_data.append(
                np.array(
                    list(product([i], sales.shop_id.unique(), sales.item_id.unique()))
                )
            )
        full_data = pd.DataFrame(np.vstack(full_data), columns=agg_group)

        return full_data.sort_values(by=agg_group)

    def prepare_full_data(
        self, items: df, categories: df, train: df, shops: df, test: df
    ) -> tuple[df, df]:
        """
        Prepare and merge data for modeling

        Returns:
        - full_data: pd.DataFrame - Complete merged data
        - train: pd.DataFrame - Processed training data
        """
        train.drop_duplicates(inplace=True)

        train["item_price"] = train["item_price"].clip(0, 50000)
        train["item_cnt_day"] = train["item_cnt_day"].clip(0, 1000)
        # Revenue feature
        train["revenue"] = train["item_price"] * train["item_cnt_day"]

        # Target creation - 'item_cnt_month'
        target_group = (
            train.groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"]
            .sum()
            .rename("item_cnt_month")
            .reset_index()
        )

        # Agg columns and periods for full_data with all shop&item pairs
        columns = ["date_block_num", "shop_id", "item_id"]
        periods = train["date_block_num"].nunique()

        full_data = self.full_data_creation(
            df=train, agg_group=columns, periods=periods
        )
        # Merge full data with target
        full_data = full_data.merge(target_group, on=columns, how="left")

        # Test set preparation and merge with full data
        test["date_block_num"] = 34
        test = test.drop(columns="ID", errors="ignore")
        # Missing filling and target clipping
        full_data = pd.concat(
            [full_data, test], keys=columns, ignore_index=True, sort=False
        )
        full_data = full_data.fillna(0)
        full_data["item_cnt_month"] = (
            full_data["item_cnt_month"].clip(0, 20).astype(np.float16)
        )

        # Encoding and feature engineering
        encoder = LabelEncoder()

        shops["city"] = (
            shops["shop_name"].str.split(" ").str[0].replace("Сергиев", "Сергиев Посад")
        )
        shops["city_id"] = encoder.fit_transform(shops["city"])

        categories["main_category"] = (
            categories["item_category_name"].str.split(" - ").apply(lambda x: x[0])
        )
        categories.replace(
            {"main_category": ["Игры PC", "Игры Android", "Игры MAC"]},
            "Игры",
            inplace=True,
        )
        categories.replace(
            {"main_category": ["Карты оплаты (Кино, Музыка, Игры)"]},
            "Карты оплаты",
            inplace=True,
        )
        categories.replace(
            {
                "main_category": [
                    "PC",
                    "Чистые носители (штучные)",
                    "Чистые носители (шпиль)",
                    "Чистые носители",
                ]
            },
            "Аксессуары",
            inplace=True,
        )
        categories.replace(
            {"main_category": ["Билеты (Цифра)", "Служебные"]}, "Билеты", inplace=True
        )
        categories["main_category_id"] = encoder.fit_transform(
            categories["main_category"]
        )
        categories["minor_category"] = (
            categories["item_category_name"]
            .str.split(" - ")
            .apply(lambda x: x[1] if len(x) > 1 else x[0])
        )
        categories["minor_category_id"] = encoder.fit_transform(
            categories["minor_category"]
        )

        # Merging full_data with all additional information from shops, items, categories dataframes
        full_data = full_data.merge(shops, on="shop_id", how="left")
        full_data = full_data.merge(items, on="item_id", how="left")
        full_data = full_data.merge(categories, on="item_category_id", how="left")
        # Also train merge
        train = train.merge(
            items.loc[:, ["item_id", "item_category_id"]], on="item_id", how="left"
        )
        train = train.merge(
            shops.loc[:, ["shop_id", "city_id"]], on="shop_id", how="left"
        )

        # Month and year features
        date_group = full_data.groupby("date_block_num").agg({"item_cnt_month": "sum"})
        date_group = date_group.reset_index()
        date_group["date"] = pd.date_range(start="2013-01-01", periods=35, freq="ME")
        date_group["month"] = date_group["date"].dt.month
        date_group["year"] = date_group["date"].dt.year
        date_group.drop(columns=["date", "item_cnt_month"], inplace=True)
        full_data = full_data.merge(date_group, on="date_block_num", how="left")

        # Column selection
        work_columns = [
            "date_block_num",
            "shop_id",
            "item_cnt_month",
            "item_id",
            "city_id",
            "item_category_id",
            "main_category_id",
            "minor_category_id",
            "year",
            "month",
        ]
        full_data = full_data.loc[:, work_columns]

        # Shop_id encoding
        full_data["shop_id"] = LabelEncoder().fit_transform(full_data["shop_id"])

        self.memory_reducer.reduce(full_data)
        self.memory_reducer.reduce(train)

        return full_data, train


class FeatureExtractor:
    def __init__(self, full_data: df, train: df, memory_reducer: MemoryReducer):
        """
        Initialize with an existing DataFrame (full_data) for feature extraction

        Parameters:
        memory_reducer: Class - to reduce memory used by dataframe
        full_data: pd.DataFrame - Pre-existing full data containing required columns
        train: pd.DataFrame - Training data for aggregating revenue-based features
        """
        self.memory_reducer = memory_reducer
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
        self.memory_reducer.reduce(self.full_data)

        return self.full_data

class MainPipeline:
    """Orchestrates the entire data pipeline"""

    def __init__(self, data_source="local", config_path="config.yaml"):
        self.loader = DataLoader(data_source=data_source, config_path=config_path)
        self.data_source = data_source
        self.config = self.loader.config 
        self.memory_reducer = MemoryReducer()
        self.preparer = DataPreparer(self.memory_reducer)

    def save_to_destination(self, df: pd.DataFrame, key: str):
        """
        Save a DataFrame either to GCS or locally, based on data_source

        Parameters:
        - df: pd.DataFrame - Data to save
        - key: str - Key in config for the save path
        """
        if self.data_source == "gcs":
            fs = gcsfs.GCSFileSystem()
            gcs_path = self.config["gcs_paths"][key]
            with fs.open(gcs_path, "w") as f:
                df.to_csv(f, index=False)
        elif self.data_source == "local":
            local_path = Path(self.config["local_paths"][key])
            local_path.parent.mkdir(parents=True, exist_ok=True) 
            df.to_csv(local_path, index=False)
        else:
            raise ValueError(f"Invalid data source: {self.data_source}")

    def run(self, test_file = None):
        # Load data
        items = self.loader.load("items")
        categories = self.loader.load("categories")
        train = self.loader.load("sales_train")
        shops = self.loader.load("shops")
        if test_file:
            try:
                test = pd.read_csv(test_file)
            except:
                raise ValueError('You should place test data in .csv format in working directory')
        else:
            test = self.loader.load('test')

        # Prepare data
        full_data, train = self.preparer.prepare_full_data(
            items, categories, train, shops, test
        )

        # Save full_data and train
        self.save_to_destination(full_data, "full_data")
        self.save_to_destination(train, "train")

        # Run feature extraction
        extractor = FeatureExtractor(
            full_data=full_data, train=train, memory_reducer=self.memory_reducer
        )
        full_featured_data = extractor.process()

        # Save extracted features
        self.save_to_destination(full_featured_data, "full_featured_data")

        print(f"Processed data saved to {self.data_source} destinations")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", required=True, help="Data source: 'local' or 'gcs'")
    args = parser.parse_args()

    pipeline = MainPipeline(data_source=args.data_source)
    pipeline.run()
