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

        X = X.fillna(0)
        
        #Merging all combinations with original dataset
        return X


#Feature engineering
class FeatureEngineering_fulldata(BaseEstimator, TransformerMixin):

    def __init__(self, target_col = 'item_cnt_month', item_price = 'item_price'):
        self.target_col = target_col
        self.item_price = item_price

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):

        #date_item_avg_item_cnt - mean sales per item per period block
        group = X.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
        group.columns = [ 'date_item_avg_item_cnt']
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['date_block_num','item_id'], how='left')
        X['date_item_avg_item_cnt'] = X['date_item_avg_item_cnt'].astype(np.float16)

        #date_shop_avg_item_cnt - mean sales per shop per period block
        group = X.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
        group.columns = [ 'date_shop_avg_item_cnt' ]
        group.reset_index(inplace=True)

        X = pd.merge(X, group, on=['date_block_num','shop_id'], how='left')
        X['date_shop_avg_item_cnt'] = X['date_shop_avg_item_cnt'].astype(np.float16)
       

        #date_shop_cat_avg_item_cnt - average sales per date_block, shop and category
        group = X.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
        group.columns = ['date_shop_cat_avg_item_cnt']
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
        X['date_shop_cat_avg_item_cnt'] = X['date_shop_cat_avg_item_cnt'].astype(np.float16)

        #Average sales per category per date_block_num
        group = X.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
        group.columns = [ 'date_cat_avg_item_cnt' ]
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['date_block_num','item_category_id'], how='left')
        X['date_cat_avg_item_cnt'] = X['date_cat_avg_item_cnt'].astype(np.float16)

        #Average sales per minor_category per date_block_num
        group = X.groupby(['date_block_num', 'minor_category_id']).agg({'item_cnt_month': ['mean']})
        group.columns = [ 'date_minor_cat_avg_item_cnt' ]
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['date_block_num','minor_category_id'], how='left')
        X['date_minor_cat_avg_item_cnt'] = X['date_minor_cat_avg_item_cnt'].astype(np.float16)

        #Average sales per main_category per date_block_num
        group = X.groupby(['date_block_num', 'main_category_id']).agg({'item_cnt_month': ['mean']})
        group.columns = [ 'date_main_cat_avg_item_cnt' ]
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['date_block_num','main_category_id'], how='left')
        X['date_main_cat_avg_item_cnt'] = X['date_main_cat_avg_item_cnt'].astype(np.float16)

        #Average sales per date block per city
        group = X.groupby(['date_block_num', 'city_id']).agg({'item_cnt_month': ['mean']})
        group.columns = [ 'date_city_avg_item_cnt' ]
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['date_block_num', 'city_id'], how='left')
        X['date_city_avg_item_cnt'] = X['date_city_avg_item_cnt'].astype(np.float16)

        #Average price per item per date_block
        group = X.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
        group.columns = ['date_item_avg_item_price']
        group.reset_index(inplace=True)

        X = pd.merge(X, group, on=['date_block_num','item_id'], how='left')
        X['date_item_avg_item_price'] = X['date_item_avg_item_price'].astype(np.float16)


        #Impact of each shop to the overall revenue
        group = X.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
        group.columns = ['date_shop_revenue']
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['date_block_num','shop_id'], how='left')
        X['date_shop_revenue'] = X['date_shop_revenue'].astype(np.float32)

        group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
        group.columns = ['shop_avg_revenue']
        group.reset_index(inplace=True)
        X = pd.merge(X, group, on=['shop_id'], how='left')
        X['shop_avg_revenue'] = X['shop_avg_revenue'].astype(np.float32)

        X['delta_revenue'] = (X['date_shop_revenue'] - X['shop_avg_revenue']) / X['shop_avg_revenue']
        X['delta_revenue'] = X['delta_revenue'].astype(np.float16)


        return X


#Lag features for target column
class LagFeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, col_lags_dict):
        self.col_lags_dict = col_lags_dict

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        
        for column, lags in self.col_lags_dict.items():
            tmp = X[['date_block_num','shop_id','item_id', column]]
            for lag in lags:
                shifted = tmp.copy()
                shifted.columns = ['date_block_num','shop_id','item_id', column+'_lag_'+str(lag)]
                shifted['date_block_num'] += lag
                X = pd.merge(X, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        return X

#Pipeline for featureengineering, lag features, obtaining full data with all shop&item pairs
def pipeline_1(date_block_num, col_lags_dict):

    pipeline_1 = Pipeline(steps = [
        ('feature engineering', FeatureEngineering()),
        ('full set', FullDataframeCreation(date_block_num)),
        ('feature engineering full_data', FeatureEngineering_fulldata()),
        ('lag features', LagFeatureGenerator(col_lags_dict = col_lags_dict))
    ])



    return pipeline_1



#Log Transformations
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, target_column):
        self.target_column = target_column
    
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        
        X[self.target_column] = np.log1p(X[self.target_column])
        
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


#Pipeline for target-log-transformation ans categorical encoding
def pipeline_2(target_log, categorical_columns):

    categorical_transformer = Pipeline(steps = [
        ('encoding', MulticolumnsLabelEncoding(columns=categorical_columns))
    ])

    log_transformer = Pipeline(steps = [
        ('log_transform', LogTransformer(target_column = target_log))
    ])


    preprocessor = ColumnTransformer(
        transformers = [
            ('log', log_transformer, target_log),
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


        
                                               


    

    


