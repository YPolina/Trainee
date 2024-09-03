import pandas as pd
from pandas.core.frame import DataFrame


class ELTPipline():

    def __init__(self):
        self.dir_ = './competitive-data-science-predict-future-sales'

    def exctraction(self, file_path: str) -> DataFrame:
        return pd.read_csv(f'{self.dir_}/{file_path}')

    def transform(self, df: DataFrame) -> DataFrame:

        if not df.isna().values.any():
            df.dropna(inplace=True)

        if df.duplicated().values.any():
            df.drop_duplicates(inplace=True)

        def outliers(self, df: DataFrame, column: str) -> DataFrame:

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            return outliers
        
        
        for column in df.columns:
            df.drop(df[df[outliers(df, column)] == True].index, inplace=True)

        return df
            


'''
def load(self, df: DataFrame, save_path: str):

    df.to_csv(save_path, index = False)

    return 
'''