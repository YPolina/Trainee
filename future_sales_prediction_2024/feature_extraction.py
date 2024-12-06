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
from future_sales_prediction_2024.data_handling import MemoryReducer


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
    extractor = FeatureExtractor(full_data=full_data, train=train, memory_reducer = MemoryReducer)
    full_featured_data = extractor.process()

    with fs.open(f"{args.outdir}/full_featured_data.csv", "w") as f:
        full_featured_data.to_csv(f, index=False)

    print(f"Full featured data saved to {args.outdir}")
