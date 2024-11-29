import pandas as pd
import gcsfs
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import product
from pandas.core.frame import DataFrame as df
import argparse


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
    



def prepare_full_data(items: df, categories: df, train: df, shops: df, test: df) -> df:
    """
    Create full data from all dataframes

    Parameters:
    - train: pd.DataFrame - The DataFrame with target value
    - test: pd.DataFrame - The DataFrame with shop&item pairs we need to make prediction
    - shops: pd.DataFrame - The DataFrame with shop_id and city_id information
    - items: pd.DataFrame - The DataFrame with item_id and item_category_id information
    - categories: pd.DataFrame - The DataFrame with item_category_id, main_category_id, minor_category_id information

    Returns:
    - full_data: pd.DataFrame - The DataFrame with merged information from all input dataframes
    """
    # Outliers drop from EDA
    train.drop(
        train[
            (
                train.item_id.isin(
                    [
                        20842,
                        21483,
                        13200,
                        5748,
                        475,
                        476,
                        3143,
                        14170,
                        1349,
                        2410,
                        7238,
                        2411,
                        3142,
                        102,
                        14173,
                        5909,
                        7241,
                        4856,
                        5960,
                    ]
                )
            )
        ].index,
        inplace=True,
    )
    train["item_price"] = train["item_price"].clip(0, 50000)
    train["item_cnt_day"] = train["item_cnt_day"].clip(0, 1000)
    train.drop_duplicates(inplace=True)

    # Target creation - 'item_cnt_month'
    target_group = (
        train.groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"]
        .sum()
        .rename("item_cnt_month")
        .reset_index()
    )

    # Revenue feature
    train["revenue"] = train["item_price"] * train["item_cnt_day"]

    # Agg columns and periods for full_data with all shop&item pairs
    columns = ["date_block_num", "shop_id", "item_id"]
    periods = train["date_block_num"].nunique()

    full_data = full_data_creation(df=train, agg_group=columns, periods=periods)

    # Merge full data with target
    full_data = full_data.merge(target_group, on=columns, how="left")

    # Test set preparation and merge with full data
    test["date_block_num"] = 34
    test = test.drop(columns="ID", errors="ignore")
    full_data = pd.concat(
        [full_data, test], keys=columns, ignore_index=True, sort=False
    )

    # Missing filling and target clipping
    full_data = full_data.fillna(0)
    full_data["item_cnt_month"] = (
        full_data["item_cnt_month"].clip(0, 20).astype(np.float16)
    )

    # City feature
    encoder = LabelEncoder()
    shops["city"] = shops["shop_name"].str.split(" ").apply(lambda x: x[0])
    shops.replace({"city": "Сергиев"}, "Сергиев Посад", inplace=True)
    shops["city_id"] = encoder.fit_transform(shops["city"])

    # Categories features
    categories["main_category"] = (
        categories["item_category_name"].str.split(" - ").apply(lambda x: x[0])
    )
    categories.replace(
        {"main_category": ["Игры PC", "Игры Android", "Игры MAC"]}, "Игры", inplace=True
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
    categories["main_category_id"] = encoder.fit_transform(categories["main_category"])
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
    train = train.merge(shops.loc[:, ["shop_id", "city_id"]], on="shop_id", how="left")

    # Month and year features
    group = full_data.groupby("date_block_num").agg({"item_cnt_month": "sum"})
    group = group.reset_index()
    group["date"] = pd.date_range(start="2013-01-01", periods=35, freq="ME")
    group["month"] = group["date"].dt.month
    group["year"] = group["date"].dt.year
    group.drop(columns=["date", "item_cnt_month"], inplace=True)
    full_data = full_data.merge(group, on="date_block_num", how="left")

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

    reduce_mem_usage(full_data)
    reduce_mem_usage(train)


    return full_data, train


# Reducing memory usage
def reduce_mem_usage(df: df, verbose: bool = True) -> None:
    """
    Reduces memory usage of a DataFrame by downcasting numeric columns to more efficient types

    Parameters:
    - df: pd.DataFrame - The DataFrame to optimize
    - verbose: bool - Whether to print memory usage reduction details (default=True)

    """

    # To divide numerical and str
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    # Iteration along columns
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            # Int dtypes
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            # Float dtypes
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
    # To get information about the progress
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return None


# Creating df with full range of data
def full_data_creation(df: df, agg_group: list, periods: int) -> df:
    """
    Generates a DataFrame with the full range of specified item and shop combinations for each period

    Parameters:
    - df: pd.DataFrame - Input DataFrame with existing data
    - agg_group: list - List of columns to aggregate (e.g., ['date_block_num', 'shop_id', 'item_id'])
    - periods: int - Number of periods to include in the generated DataFrame

    Returns:
    - pd.DataFrame - DataFrame containing all combinations of items, shops, and periods
    """

    full_data = []

    # Iteration along all date blocks
    for i in range(periods):
        sales = df[df.date_block_num == i]
        # Adding all possible combinations item_id&shop_id
        full_data.append(
            np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())))
        )

    full_data = pd.DataFrame(np.vstack(full_data), columns=agg_group)
    full_data = full_data.sort_values(by=agg_group)

    return full_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="Path to items.csv in GCS")
    parser.add_argument("--categories", required=True, help="Path to item_categories.csv in GCS")
    parser.add_argument("--train", required=True, help="Path to sales_train.csv in GCS")
    parser.add_argument("--shops", required=True, help="Path to shops.csv in GCS")
    parser.add_argument("--test", required=True, help="Path to test.csv in GCS")
    parser.add_argument("--outdir", required=True, help="Path in GCS to save processed data")
    args = parser.parse_args()

    fs = gcsfs.GCSFileSystem()

    # Load data using the loader
    items_df = loader(args.items)
    categories_df = loader(args.categories)
    train_df = loader(args.train)
    shops_df = loader(args.shops)
    test_df = loader(args.test)

    # Prepare full data and train
    full_data, train = prepare_full_data(items_df, categories_df, train_df, shops_df, test_df)

    with fs.open(f"{args.outdir}/full_data.csv", "w") as f:
        full_data.to_csv(f, index=False)

    with fs.open(f"{args.outdir}/train.csv", "w") as f:
        train.to_csv(f, index=False)

    print(f"Processed data saved to {args.outdir}")