import pandas as pd
from pandas.core.frame import DataFrame
from pandas import Series


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


    def outliers(self, df: DataFrame, group: str, val: str) -> DataFrame:

        qs = df.groupby(group)[val].quantile([0.25,0.75])
        qs = qs.unstack().reset_index()
        qs.columns = [f'{group}', "q1", "q3"]
        df_m = pd.merge(df, qs, on=f'{group}', how="left")
        df_m["Outlier"] = ~ df_m[val].between(df_m["q1"], df_m["q3"])
        return df_m[df_m['Outlier'] == False].drop(['q1', 'q3', 'Outlier'], axis = 1)
        
        
    
    def data_format(self, data: Series) -> Series:

        return pd.to_datetime(data, format = '%d.%m.%Y')
            
    def merge(self):
        return df

'''
def load(self, df: DataFrame, save_path: str):

    df.to_csv(save_path, index = False)

    return 
'''