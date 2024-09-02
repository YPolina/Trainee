import pandas as pd
from pandas.core.frame import DataFrame

def exctract(file_path: str) -> DataFrame:

    df = pd.read_csv(file_path)

    return df

def transform(df: DataFrame) -> DataFrame:

    df.dropna(inplace= True)

    return df

def load(df: DataFrame, save_path: str):

    df.to_csv(save_path, index = False)

    return
