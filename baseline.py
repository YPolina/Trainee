import pandas as pd
import os
from pandas.core.frame import DataFrame as df
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import TypeVar
from sklearn.preprocessing import LabelEncoder
from itertools import product
from typing import Union
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

#To use in annotation for the self parameter of class Validator
TValidator = TypeVar("TValidator", bound="Validator")

def loader(directory: str = './competitive-data-science-predict-future-sales') -> df:
    '''
    Data loader from .csv

    Parameters:
    - directory: str - where are files are stored

    Returns:
    data: pd.DataFrames - data from .csv file
    '''

    file_paths = os.listdir(directory)
    dataframes = [pd.read_csv(os.path.join(directory, path)) for path in file_paths]
    del dataframes[3]
    return dataframes

def prepare_full_data(items: df, categories: df, train: df, shops:df,  test: df) -> df:
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
    #Outliers drop from EDA 
    train.drop(train[(train.item_id.isin([20842, 21483, 13200, 5748,   475,   476,  3143, 14170,  1349,  2410,  7238,  2411,
        3142,   102, 14173,  5909,  7241,  4856, 5960]))].index, inplace = True)
    train['item_price'] = train['item_price'].clip(0, 50000)
    train['item_cnt_day'] = train['item_cnt_day'].clip(0, 1000)
    train.drop_duplicates(inplace = True)

    #Target creation - 'item_cnt_month'
    target_group = (
        train.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day']
        .sum().rename('item_cnt_month').reset_index()
    )
    
    #Revenue feature
    train['revenue'] = train['item_price'] * train['item_cnt_day']
    
    #Agg columns and periods for full_data with all shop&item pairs
    columns = ['date_block_num', 'shop_id', 'item_id']
    periods = train['date_block_num'].nunique()

    
    full_data = full_data_creation(df=train, agg_group=columns, periods=periods)
    
    #Merge full data with target
    full_data = full_data.merge(target_group, on=columns, how='left')
    
    #Test set preparation and merge with full data
    test['date_block_num'] = 34
    test = test.drop(columns='ID', errors='ignore')
    full_data = pd.concat([full_data, test], keys=columns, ignore_index=True, sort=False)
    
    #Missing filling and target clipping
    full_data = full_data.fillna(0)
    full_data['item_cnt_month'] = full_data['item_cnt_month'].clip(0, 20).astype(np.float16)

    #City feature
    encoder = LabelEncoder()
    shops['city'] = shops['shop_name'].str.split(' ').apply(lambda x: x[0])
    shops.replace({'city': 'Сергиев'}, 'Сергиев Посад', inplace=True)
    shops['city_id'] = encoder.fit_transform(shops['city'])

    #Categories features
    categories['main_category'] = categories['item_category_name'].str.split(' - ').apply(lambda x: x[0])
    categories.replace({'main_category': ['Игры PC', 'Игры Android', 'Игры MAC']}, 'Игры', inplace=True)
    categories.replace({'main_category': ['Карты оплаты (Кино, Музыка, Игры)']}, 'Карты оплаты', inplace=True)
    categories.replace({'main_category': ["PC", 'Чистые носители (штучные)', "Чистые носители (шпиль)", 'Чистые носители']}, 'Аксессуары', inplace=True)
    categories.replace({'main_category': ["Билеты (Цифра)", 'Служебные']}, 'Билеты', inplace=True)
    categories['main_category_id'] = encoder.fit_transform(categories['main_category'])
    categories['minor_category'] = categories['item_category_name'].str.split(' - ').apply(lambda x: x[1] if len(x) > 1 else x[0])
    categories['minor_category_id'] = encoder.fit_transform(categories['minor_category'])
    
    #Merging full_data with all additional information from shops, items, categories dataframes
    full_data = full_data.merge(shops, on='shop_id', how='left')
    full_data = full_data.merge(items, on='item_id', how='left')
    full_data = full_data.merge(categories, on='item_category_id', how='left')
    #Also train merge
    train = train.merge(items.loc[:, ['item_id', 'item_category_id']], on = 'item_id', how = 'left')
    train = train.merge(shops.loc[:, ['shop_id', 'city_id']], on = 'shop_id', how = 'left')

    #Month and year features
    group = full_data.groupby('date_block_num').agg({'item_cnt_month': 'sum'})
    group = group.reset_index()
    group['date'] = pd.date_range(start='2013-01-01', periods=35, freq='ME')
    group['month'] = group['date'].dt.month
    group['year'] = group['date'].dt.year
    group.drop(columns = ['date', 'item_cnt_month'], inplace = True)
    full_data = full_data.merge(group, on = 'date_block_num', how = 'left')
    
    #Column selection
    work_columns = [
        'date_block_num', 'shop_id', 'item_cnt_month', 'item_id', 
        'city_id', 'item_category_id', 'main_category_id', 'minor_category_id',
        'year', 'month'
    ]
    full_data = full_data.loc[:, work_columns]
    
    #Shop_id encoding
    full_data['shop_id'] = LabelEncoder().fit_transform(full_data['shop_id'])
    
    return full_data, train

#Class for validation
class Validator(BaseEstimator, TransformerMixin):

    """
    Initializes Validator with specified column types, value ranges, and options for checking duplicates and missing values

    Parameters:
    - column_types: Dict[str, str] - Expected data types for each column (e.g., {'shop_id': 'int64'})
    - value_ranges: Dict[str, Tuple[float, float]] - Expected numeric range for each column (e.g., {'month': (1, 12)})
    - check_duplicates: bool - Whether to check for duplicate rows in the DataFrame (default=True)
    - check_missing: bool - Whether to check for missing values in the DataFrame (default=True)
    """
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

        """
        Checks data types of DataFrame columns against expected types defined in column_types

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate column data types

        Raises:
        - TypeError if any column's data type does not match the expected type
        - ValueError if a required column is missing from the DataFrame
        """
        
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

        """
        Validates that numeric columns fall within specified minimum and maximum ranges in value_ranges

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate column value ranges

        Raises:
        - ValueError if any value in a specified column is out of range
        """
        
        #Iteration along columns
        for column, (min_value, max_value) in self.value_ranges.items():
            #If any values out of range
            if (X[column] < min_value).any() or (X[column] > max_value).any():
                raise ValueError(f'Values of column {column} are out of expected value range {min_value}-{max_value} ')
    
    def _check_non_negative_values(self: TValidator, X: df) -> Exception | None:

        """
        Checks if all values in numeric columns are non-negative

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for non-negative values

        Raises:
        - ValueError if any column contains negative values
        """
        #Iteration along columns
        for column in X.columns:
            #Negative values detection
            if (X[column] < 0).any():
                raise ValueError(f'Column {column} contains negative values')

    def _check_duplicates(self: TValidator, X: df) -> Exception | None:
    
        """
        Detects and raises an error if there are duplicate rows in the DataFrame

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for duplicates

        Raises:
        - ValueError if duplicates are found
        """
        #If duplicated columns are founded
        if X.duplicated().sum() != 0:
            raise ValueError('Duplicated rows are detected')

    def _check_missing(self: TValidator, X: df) -> Exception | None:
        """
        Detects missing values in columns of the DataFrame

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for missing values

        Raises:
        - ValueError if any column contains missing values
        """
        missing_columns = X.columns[X.isna().any()].tolist()
        #If missing columns are founded
        if missing_columns:
            raise ValueError(f'Columns {missing_columns} contain missing values')


    def fit(self: TValidator, X: df) -> TValidator:
        """
        Fits the Validator by setting column types and value ranges if not specified, based on the provided DataFrame

        Parameters:
        - X: pd.DataFrame - The DataFrame from which to infer column types and value ranges if not predefined

        Returns:
        - Validator - Returns self after setting inferred properties
        """
        if self.column_types is None:
            self.column_types = X.dtypes.to_dict()
        if self.value_ranges is None:
            self.value_ranges = {col: (X[col].min(), X[col].max()) for col in X.columns}

        return self
    
    #validation
    def transform(self: TValidator, X: df) -> str:

        """
        Performs validation checks on a DataFrame, ensuring data types, ranges, and constraints are respected

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate

        Returns:
        - str - Confirmation message ("Data is valid") if validation passes
        """

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
def reduce_mem_usage(df: df, verbose: bool=True) -> None:

    """
    Reduces memory usage of a DataFrame by downcasting numeric columns to more efficient types

    Parameters:
    - df: pd.DataFrame - The DataFrame to optimize
    - verbose: bool - Whether to print memory usage reduction details (default=True)

    """

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
    return None

#Creating df with full range of data
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

    #Iteration along all date blocks
    for i in range(periods):
        sales = df[df.date_block_num == i]
        #Adding all possible combinations item_id&shop_id
        full_data.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique()))))

    full_data = pd.DataFrame(np.vstack(full_data), columns = agg_group)
    full_data = full_data.sort_values(by = agg_group)

    return full_data


class FeatureExtractor:
    def __init__(self, full_data: df, train: df):
        """
        Initialize with an existing DataFrame (full_data) for feature extraction
        
        Parameters:
        full_data: pd.DataFrame - Pre-existing full data containing required columns
        train: pd.DataFrame - Training data for aggregating revenue-based features
        """
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
        group = (self.full_data[self.full_data.item_cnt_month > 0]
                 .groupby(agg)['date_block_num']
                 .nunique()
                 .rename(new_feature)
                 .reset_index())
        self.full_data = self.full_data.merge(group, on=agg, how='left')

    def feat_from_agg(self, df: df, agg: list, new_col: str, aggregation:list) -> df:        
        """
        Aggregates features based on specified columns, aggregation functions, and adds the result as a new feature.

        Parameters:
        - agg: list - Columns to group by (e.g., ['shop_id', 'item_id']).
        - new_col: str - Name for the new aggregated feature.
        - aggregation: Dict[str, Union[str, List[str]]] - Aggregation functions to apply on the grouped data

        Returns:
        - pd.DataFrame - DataFrame with the new aggregated feature.
        """
        temp = df[df.item_cnt_month > 0] if new_col == 'first_sales_date_block' else df.copy()
        temp = temp.groupby(agg).agg(aggregation)
        temp.columns = [new_col]
        temp.reset_index(inplace=True)
        self.full_data = pd.merge(self.full_data, temp, on=agg, how='left')
        
        if new_col == 'first_sales_date_block':
            self.full_data.fillna(34, inplace=True)

    def lag_features(self, col:str, lags: list) -> df:
        """
        Adds lagged features to the DataFrame for specified columns over defined lag periods.

        Parameters:
        - col: str - Column to create lags for.
        - lags: list - List of lag periods to apply.

        Returns:
        - pd.DataFrame - DataFrame with the newly created lagged features.
        """
        temp = self.full_data[['date_block_num', 'shop_id', 'item_id', col]]
        for lag in lags:
            shifted = temp.copy()
            shifted.columns = ['date_block_num', 'shop_id', 'item_id', f"{col}_lag_{lag}"]
            shifted['date_block_num'] += lag
            self.full_data = pd.merge(self.full_data, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')

    def new_items(self, agg: list, new_col: str) -> df:
        """
        Adds a feature tracking average monthly sales for items with specific historical conditions (e.g., item history of 1).

        Parameters:
        - agg: list - Columns to group by (e.g., ['shop_id', 'item_id']).
        - new_col: str - Name for the new column.

        Returns:
        - pd.DataFrame - DataFrame with the new column based on items' sales history.
        """

        temp = (self.full_data.query('item_history == 1')
                .groupby(agg)['item_cnt_month']
                .mean()
                .reset_index()
                .rename(columns={'item_cnt_month': new_col}))
        self.full_data = self.full_data.merge(temp, on=agg, how='left')

    def add_revenue_features(self):
        """Add revenue-based features and lags

        Returns:
        - pd.DataFrame - DataFrame with revenue lags.
        """
        # Revenue-based aggregations
        revenue_agg_list = [
            (self.train, ['date_block_num', 'item_category_id', 'shop_id'], 'sales_per_category_per_shop', {'revenue': 'sum'}),
            (self.train, ['date_block_num', 'shop_id'], 'sales_per_shop', {'revenue': 'sum'}),
            (self.train, ['date_block_num', 'item_id'], 'sales_per_item', {'revenue': 'sum'}),
        ]
        for df, agg, new_col, aggregation in revenue_agg_list:
            self.feat_from_agg(df, agg, new_col, aggregation)
        
        
        # Lag features for revenue aggregations
        revenue_lag_dict = {
            'sales_per_category_per_shop': [1],
            'sales_per_shop': [1],
            'sales_per_item': [1]
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
        self.feat_from_agg(self.train, ['shop_id'], 'avg_sales_per_shop', {'revenue': 'mean'})
        self.full_data['avg_sales_per_shop'] = self.full_data['avg_sales_per_shop'].astype(np.float32)
        self.full_data['delta_revenue_lag_1'] = (
            (self.full_data['sales_per_shop_lag_1'] - self.full_data['avg_sales_per_shop'])
            / self.full_data['avg_sales_per_shop']
        )
        self.full_data.drop(columns=['avg_sales_per_shop', 'sales_per_shop_lag_1'], inplace=True)

        # Average item price features
        self.feat_from_agg(self.train, ['item_id'], 'item_avg_item_price', {'item_price': 'mean'})
        self.full_data['item_avg_item_price'] = self.full_data['item_avg_item_price'].astype(np.float16)

        self.feat_from_agg(self.train, ['date_block_num', 'item_id'], 'date_item_avg_item_price', {'item_price': 'mean'})
        self.full_data['date_item_avg_item_price'] = self.full_data['date_item_avg_item_price'].astype(np.float16)

        # Lag for item price feature and delta price calculation
        self.lag_features('date_item_avg_item_price', [1])
        self.full_data['delta_price_lag_1'] = (
            (self.full_data['date_item_avg_item_price_lag_1'] - self.full_data['item_avg_item_price'])
            / self.full_data['item_avg_item_price']
        )
        self.full_data.drop(columns=['item_avg_item_price', 'date_item_avg_item_price', 'date_item_avg_item_price_lag_1'], inplace=True)

    def process(self):
        """Execute feature extraction on full_data

        Returns:
        - pd.DataFrame - full data with all features
        """
        # History features
        history = [
            ('shop_id', 'shop_history'),
            ('item_id', 'item_history'),
            ('minor_category_id', 'minor_category_history')
        ]
        for group, new_feature in history:
            self.history_features([group], new_feature)

        # Features from aggregations
        agg_list = [
            (self.full_data, ['date_block_num', 'item_category_id'], 'avg_item_cnt_per_cat', {'item_cnt_month': 'mean'}),
            (self.full_data, ['date_block_num', 'city_id', 'shop_id'], 'avg_item_cnt_per_city_per_shop', {'item_cnt_month': 'mean'}),
            (self.full_data, ['date_block_num', 'shop_id'], 'avg_item_cnt_per_shop', {'item_cnt_month': 'mean'}),
            (self.full_data, ['date_block_num', 'item_category_id', 'shop_id'], 'avg_item_cnt_per_cat_per_shop', {'item_cnt_month': 'mean'}),
            (self.full_data, ['date_block_num', 'item_id'], 'avg_item_cnt_per_item', {'item_cnt_month': 'mean'}),
            (self.full_data, ['date_block_num', 'item_category_id', 'shop_id'], 'med_item_cnt_per_cat_per_shop', {'item_cnt_month': 'median'}),
            (self.full_data, ['date_block_num', 'main_category_id'], 'avg_item_cnt_per_main_cat', {'item_cnt_month': 'mean'}),
            (self.full_data, ['date_block_num', 'minor_category_id'], 'avg_item_cnt_per_minor_cat', {'item_cnt_month': 'mean'}),
            (self.full_data, ['item_id'], 'first_sales_date_block', {'item_cnt_month': 'min'})
        ]
        for df, agg, new_col, aggregation in agg_list:
            self.feat_from_agg(df, agg, new_col, aggregation)

        # Lagged features
        lag_dict = {'avg_item_cnt_per_cat': [1], 'avg_item_cnt_per_shop': [1,3,6], 'avg_item_cnt_per_item': [1,3,6],
            'avg_item_cnt_per_city_per_shop': [1], 'avg_item_cnt_per_cat_per_shop': [1], 
            'med_item_cnt_per_cat_per_shop': [1], 'avg_item_cnt_per_main_cat': [1],
            'avg_item_cnt_per_minor_cat': [1], 'item_cnt_month': [1,2,3,6,12]}

        for feature, lags in lag_dict.items():
            self.lag_features(feature, lags)
            if feature != 'item_cnt_month':
                self.full_data.drop(columns=[feature], inplace=True)

        # Revenue and item price-related features
        self.add_revenue_features()
        self.add_item_price_features()

        # Last sale and time since last sale features
        self.full_data['last_sale'] = self.full_data.groupby(['shop_id', 'item_id'])['date_block_num'].shift(1)
        self.full_data['months_from_last_sale'] = self.full_data['date_block_num'] - self.full_data['last_sale']
        self.full_data['months_from_first_sale'] = self.full_data['date_block_num'] - self.full_data.groupby(['shop_id', 'item_id'])['date_block_num'].transform('min')
        self.full_data['months_from_last_sale'].fillna(-1, inplace=True)
        self.full_data.drop('last_sale', axis = 1, inplace = True)
        # Fill NaNs
        self.full_data.fillna(0, inplace=True)

        return self.full_data


#Data validation using TSS 
def tss_cv(df: df, n_splits, model: Union[LinearRegression, XGBRegressor, LGBMRegressor], true_pred_plot: bool = True):
    """
    Performs cross-validation for time series data using specified regression model and calculates RMSE.

    Parameters:
    - df: pd.DataFrame - DataFrame with features and target variable.
    - n_splits: int - Number of cross-validation splits.
    - model: Union[LinearRegression, XGBRegressor, LGBMRegressor] - Model to use for prediction.

    Returns:
    - Tuple[np.ndarray, np.ndarray, Union[LinearRegression, XGBRegressor, LGBMRegressor]] - True and predicted values, and trained model.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    n = 0
    rmse = []

    X_test = df[df.date_block_num == 34].drop('item_cnt_month', axis = 1)

    X = df[df.date_block_num != 34].drop('item_cnt_month', axis = 1)
    y = df[df.date_block_num != 34]['item_cnt_month']

    print(f'{type(model).__name__}')
    model = model

    for train_idxs, val_idxs in tss.split(X):

        X_train, X_val = X.iloc[train_idxs], X.iloc[val_idxs]
        y_train, y_val = y.iloc[train_idxs], y.iloc[val_idxs]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val).clip(0,20)
        rmse.append(root_mean_squared_error(y_pred, y_val))
        print(f'RMSE for split {n+1}: {rmse[n]:.3f}')
        n += 1
    print(f'Mean RMSE for all splits: {np.mean(rmse):.3f}')

    #Plots true versus predicted values to assess model performance visually
    if true_pred_plot:

        plt.figure(figsize = (10,6))

        #Difference
        sns.scatterplot(x = y_val, y = y_pred, color = 'blue', alpha = 0.5, s = 30, edgecolor = 'k')
        #If prediction will be equal to our target
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')

        plt.title("True vs Predicted Values")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.show()




#Split begore model fitting (with eval_set)
def data_split(df: df) -> np.ndarray:
    """
    Splits data into training, validation, and test sets for model evaluation.

    Parameters:
    - df: pd.DataFrame - DataFrame containing features and target variable.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] - Training, validation, and test sets for features and target.
    """

    X_train = df[~df.date_block_num.isin([33,34])]
    y_train = X_train['item_cnt_month']
    del X_train['item_cnt_month']

    X_val = df[df['date_block_num']==33]
    y_val = X_val['item_cnt_month']
    del X_val['item_cnt_month']

    X_test = df[df['date_block_num']==34].drop(columns='item_cnt_month')
    X_test = X_test.reset_index()
    del X_test['index']

    return X_train, y_train, X_val, y_val, X_test

#Training
def train_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, 
                  model_: Union[XGBRegressor, LGBMRegressor], model_params: dict = None, X_val: np.ndarray = None, 
                  y_val: np.ndarray = None) -> np.ndarray:
    """
    Train model and make predictions

    Parameters:
    - X_train: np.ndarray - Feature set for training
    - y_train: np.ndarray - target for training
    - X_val: np.ndarray - Feature set for validation
    - y_val: np.ndarray - target for validation
    - X_test: np.ndarray - Feature set for prediction
    - model: Union[LinearRegression, XGBRegressor, LGBMRegressor] - Trained model to use for predictions
    - model_params: Dict[str, Any] - Model parameters to be set using set_params

    Returns:
    - y_pred: np.ndarray - prediction
    """
    model = model_
    if model_params is not None:
        model.set_params(**model_params)

    if isinstance(model, LinearRegression):
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Make predictions
    y_pred = np.round(model.predict(X_test),2).clip(0,20)

    return y_pred



#Prediction creating submission file
def submission(y_pred: np.ndarray) -> str:
    """
    Saves a submission file with predictions for the test set.

    Parameters:
    - y_pred: np.ndarray - prediction

    Returns:
    - str - Confirmation message indicating submission file was created
    """

    submission = pd.DataFrame({'ID': np.arange(len(y_pred)), 'item_cnt_month': y_pred})
    submission.to_csv('submission.csv', index = False)
    return 'Submission file created'



#Feature importances plot
def feature_importances_plot(X_train: np.ndarray, model: Union[XGBRegressor, LGBMRegressor]) -> Figure:

    """
    Plots the feature importance scores for a trained model.

    Parameters:
    - X_train: np.ndarray - Training feature set.
    - model: Union[XGBRegressor, LGBMRegressor] - Trained model with feature importance scores.

    Returns:
    - Figure - Horizontal bar plot of feature importances.
    """

    plt.figure(figsize = (15,15))
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx], color = 'springgreen')
    plt.ylabel('Features')
    plt.xlabel('F-score')
    plt.title(f'{type(model).__name__} Feature Importances')
    plt.show()





