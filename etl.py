import pandas as pd
from pandas.core.frame import DataFrame

dir_ = './competitive-data-science-predict-future-sales'


def exctraction(file_path: str) -> DataFrame:
    return pd.read_csv(f'{dir_}/{file_path}')

def transform(self, df: DataFrame) -> DataFrame:

    df.dropna(inplace= True)

    return df


'''
def load(self, df: DataFrame, save_path: str):

    df.to_csv(save_path, index = False)

    return 
'''