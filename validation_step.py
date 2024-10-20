import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#Feature engineering
class FeatureEngineering():

    def __init__(self, target_col = 'item_cnt_month', item_price = 'item_price'):
        self.target_col = target_col
        self.item_price = item_price

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):

        #Revenue - item_cnt_month * item_price
        X['revenue'] = X[self.target_col] * X[self.item_price]

        #Shop history: To track amount of time blocks with data for each shop
        X['shop_history'] = (X.groupby('shop_id')['date_block_num'].transform('nunique'))
        #Minor_category_history: To track amount of time blocks with data for each minor_category
        X['minor_category_history'] = (X.groupby('minor_category_id')['date_block_num'].transform('nunique'))

        return X

#To ensure that our df has all shop&item combinations
class FullDataframeCreation(BaseEstimator, TransformerMixin):

    def __init__(self, date_block_num):
        self.date_block_num = date_block_num

    def fit(self, X, y = None):

        return self

    def transform(self, X):

        #All unique values of existed shops and items
        full_data_list = []
        shop_ids = X['shop_id'].unique()
        item_ids = X['item_id'].unique()
        
        #Iterating through unique values to get all shop&item combinations
        for i in range(self.date_block_num + 1):
            for shop in shop_ids:
                for item in item_ids:
                    full_data_list.append([i, shop, item])
        columns = ['date_block_num','shop_id','item_id']
        full_data = pd.DataFrame(full_data_list, columns=columns)
        full_data.sort_values(by = columns, inplace = True)
        
        X = pd.merge(full_data, X, on = columns, how = 'left')
        
        #Merging all combinations with original dataset
        return X


#Lag features for target column
class LagFeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, target_col = 'item_cnt_month', group_cols = ['date_block_num', 'shop_id', 'item_id'], lags = [1,2,3,6,12]):
        self.lags = lags
        self.group_cols = group_cols
        self.target_col = target_col

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        
        
        for lag in self.lags:
            X[f'{self.target_col}_lag_{lag}'] = X.groupby(self.group_cols)[self.target_col].shift(lag)
        X = X.fillna(0)
        return X

def pipeline_1(date_block_num, lags):

    pipeline_1 = Pipeline(steps = [
        ('feature engineering', FeatureEngineering()),
        ('full set', FullDataframeCreation(date_block_num)),
        ('lag features', LagFeatureGenerator(lags = lags))
    ])



    return pipeline_1


#Log Transformations
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        
        for col in self.columns:
            X[col] = np.log1p(X[col])
        
        return X



#To handle encoding column by column
class MulticolumnsLabelEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, columns = None):
        self.columns = columns
        #To store encoders for each column
        self.encoders = {}

    #Fit each column separatly
    def fit(self, X, y = None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        
        return self
    
    #Apply Label Encoder for each column
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)
        
        for col in self.columns:
            X[col] = self.encoders[col].transform(X[col].astype(str))
        
        return X


#All steps
def pipeline_2(log_transform_cols, categorical_columns):

    categorical_transformer = Pipeline(steps = [
        ('encoding', MulticolumnsLabelEncoding(columns=categorical_columns))
    ])

    log_transformer = Pipeline(steps = [
        ('log_transform', LogTransformer(columns = log_transform_cols))
    ])


    preprocessor = ColumnTransformer(
        transformers = [
            ('log', log_transformer, log_transform_cols),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    pipeline_2 = Pipeline(steps = [
        ('preprocessor', preprocessor)
    ])



    return pipeline_2



#Validation
class Validator(BaseEstimator, TransformerMixin):

    
    def __init__(self, column_types = None, value_ranges = None, non_negative_columns = None, check_duplicates = True, check_missing = True):
        #column types:dict
        #Expected data type for each column {'shop_id': 'int64'}
        self.column_types = column_types
        #value ranges:dict
        #Expected value range for numeric columns {'month' : (1, 12)}
        self.value_ranges = value_ranges
        #non_negative_values:list
        #List of columns that should be not negative
        self.non_negative_columns = non_negative_columns
        #check_missing: bool
        #Whether to check missing values in dataset
        self.check_missing = check_missing
        #check_duplicates:bool
        #Wether to check duplicates in dataset
        self.check_duplicates = check_duplicates


    #Check of data dtype
    def _check_column_types(self, X):

        #Iteration through all column types
        for col, expected_column_type in self.column_types.items():
            
            if col in X.columns:
                if not pd.api.types.is_dtype_equal(X[col].dtype, np.dtype(expected_column_type)):
                    raise TypeError(f'Column {col} should of type {expected_column_type}, provided {X[col].dtype} type')
            #If we do not have expected column in X
            else:
                raise ValueError(f'There is no column {col} in dataset')

    
    
    #Check data range
    def _check_value_ranges(self, X):
        
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in X.columns:
                if (X[col] < min_val).any() or (X[col] > max_val).any():
                    raise ValueError(f'Column {col} is outside the range ({min_val}, {max_val})')
            else:
                raise ValueError(f'There is no column {col} in dataset')


    #Check of positive values
    def _check_non_negative_values(self, X):

        for col in self.non_negative_columns:
            if col in X.columns:
                if (X[col] < 0).any():
                    raise ValueError(f'Column {col} contains negative values')
            else:
                raise ValueError(f'There is no column {col} in dataset')
        
    #Check missing values 
    def _check_missing_values(self, X):
        
        missing_columns = X.columns[X.isna().any()].tolist()
        if missing_columns:
            raise ValueError(f'The following columns have missing values: {missing_columns}')
        
    #check duplicates
    def _check_duplicates(self, X):
        
        if X.duplicated().any():
            raise ValueError('The dataset contains duplicated rows')

    
    #Fitting model
    def fit(self, X, y = None):

        if self.column_types is None:
            self.column_types = X.dtypes.to_dict()
        if self.value_ranges is None:
            self.value_ranges = {col: (X[col].min(), X[col].max()) for col in X.select_dtypes(include = [np.number])}
        if self.non_negative_columns is None:
            self.non_negative_columns = [col for col in X.select_dtypes(include = [np.number])]

        return self

    #Validation checks
    def transform(self, X):

        if self.column_types:
            self._check_column_types(X)

        if self.value_ranges:
            self._check_value_ranges(X)

        if self.non_negative_columns:
            self._check_non_negative_values(X)

        if self.check_duplicates:
            self._check_duplicates(X)
        
        if self.check_missing:
            self._check_missing_values(X)
        
        return X


        
                                               


    

    

