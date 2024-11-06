import pandas as pd
from pandas.core.frame import DataFrame as df
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import TypeVar
from itertools import product

#To use in annotation for the self parameter of class Validator
TValidator = TypeVar("TValidator", bound="Validator")

#Class for validation
class Validator(BaseEstimator, TransformerMixin):

    def __init__(self: TValidator, column_types: dict, value_ranges:dict, check_duplicates:bool = True, check_missing:bool = True) -> None:

        #Expected data type for each column {'shop_id': 'int64'}
        self.column_types: dict = column_types
        #Expected value range for numeric columns {'month' : (1, 12)}
        self.value_ranges: dict = value_ranges
        #Whether to check duplicates in data
        self.check_duplicates: bool = check_duplicates
        #Whether to check missing values in data
        self.check_missing: bool = check_missing

    #Column types
    def _check_dtypes(self: TValidator, X: df) -> Exception | None:
        
        #Iteration through all columns
        for column, expected_column_type in self.column_types.items():
            #If we have column in data
            if column in X.columns:
                #Check the equality of dtypes
                if not pd.api.types.is_dtype_equal(X[column].dtype, np.dtype(expected_column_type)):
                    raise TypeError(f'For column {column} expected {expected_column_type} dtype')
            #If do not have expected column
            else:
                raise ValueError(f'Expected {column} not found')
    
    def _check_value_ranges(self: TValidator, X: df) -> Exception | None:
        
        #Iteration along columns
        for column, (min_value, max_value) in self.value_ranges.items():
            #If any values out of range
            if (X[column] < min_value).any() or (X[column] > max_value).any():
                raise ValueError(f'Values of column {column} are out of expected value range {min_value}-{max_value} ')
    
    def _check_non_negative_values(self: TValidator, X: df) -> Exception | None:

        #Iteration along columns
        for column in X.columns:
            #Negative values detection
            if (X[column] < 0).any():
                raise ValueError(f'Column {column} contains negative values')

    def _check_duplicates(self: TValidator, X: df) -> Exception | None:
        #If duplicated columns are founded
        if X.duplicated().sum() != 0:
            raise ValueError('Duplicated rows are detected')

    def _check_missing(self: TValidator, X: df) -> Exception | None:
        
        missing_columns = X.columns[X.isna().any()].tolist()
        #If missing columns are founded
        if missing_columns:
            raise ValueError(f'Columns {missing_columns} contain missing values')


    def fit(self: TValidator, X: df) -> TValidator:

        if self.column_types is None:
            self.column_types = X.dtypes.to_dict()
        if self.value_ranges is None:
            self.value_ranges = {col: (X[col].min(), X[col].max()) for col in X.columns}

        return self
    
    #validation
    def transform(self: TValidator, X: df) -> str:

        if self.column_types:
            self._check_dtypes(X)
        if self.value_ranges:
            self._check_value_ranges(X)
            self._check_non_negative_values(X)
        if self.check_missing:
            self._check_missing
        if self.check_duplicates:
            self._check_duplicates(X)

        return f'Data is valid'

#Reducing memory usage
def reduce_mem_usage(df: df, verbose: bool=True) -> df:

    #To divide numerical and str
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2   
    #Iteration along columns 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            #Int dtypes
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            #Float dtypes  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    #To get information about the progress
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#Creating df with full range of data
def full_data_creation(df: df, agg_group: list, periods: int) -> df:

    full_data = []

    #Iteration along all date blocks
    for i in range(periods):
        sales = df[df.date_block_num == i]
        #Adding all possible combinations item_id&shop_id
        full_data.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique()))))

    full_data = pd.DataFrame(np.vstack(full_data), columns = agg_group)
    full_data = full_data.sort_values(by = agg_group)

    return full_data


def history_features(df: df, agg: list, new_feature: str) -> df:

    group = (df[df.item_cnt_month > 0].groupby(agg)
            ['date_block_num'].nunique().rename(new_feature).reset_index())

    return df.merge(group, on = agg, how = 'left')

def feat_from_agg(df: df, agg: list, new_col: str, aggregation: str) -> df:
    
    if new_col == 'first_sales_date_block':
        temp = df[df.item_cnt_month > 0]
    else:
        temp = df.copy()
    temp = temp.groupby(agg).agg(aggregation).reset_index()
    temp.columns = agg + [new_col]
    df = pd.merge(df, temp, on=agg, how='left')

    return df

#Lags
def lag_features(df: df, shift_col: str, group_columns: list = ['shop_id', 'item_id'], lags: list = [1,2,3]) -> df:
    for lag in lags:
        new_col = '{0}_lag_{1}'.format(shift_col, lag)
        df[new_col] = df.groupby(group_columns)[shift_col].shift(lag)
    if shift_col != 'item_cnt_month':
        del df[shift_col]
    return df

def new_items(df: df, agg: list, new_col: str) -> df:
    #Track of items with history == 1

    temp = (df.query('item_history == 1')
    .groupby(agg)
    .agg({'item_cnt_month': 'mean'})
    .reset_index()
    .rename(columns = {'item_cnt_month':new_col}))

    return pd.merge(df, temp, on = agg, how = 'left')


def last_sales(df: df, new_feature: str, item_shop: bool) -> df:

    cache = {}
    #Feature initialization
    df[new_feature] = -1
    df[new_feature] = df[new_feature].astype(np.int8)

    for idx, row in df.iterrows():
        #If we need to agg item_id+shop_id
        if item_shop:
             key = str(row.item_id)+' '+str(row.shop_id)
        else:
            key = row.item_id
        #First "last_sale" detection
        if key not in cache:
            if row.item_cnt_month!=0:
                cache[key] = row.date_block_num
        #If we have already sales
        else:
            last_date_block_num = cache[key]
            df.at[idx, new_feature] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num 

    return df
  
