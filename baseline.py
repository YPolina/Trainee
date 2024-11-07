import pandas as pd
from pandas.core.frame import DataFrame as df
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import TypeVar
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

#Class for validation
class Validator(BaseEstimator, TransformerMixin):

    """
    Initializes Validator with specified column types, value ranges, and options for checking duplicates and missing values.

    Parameters:
    - column_types: Dict[str, str] - Expected data types for each column (e.g., {'shop_id': 'int64'}).
    - value_ranges: Dict[str, Tuple[float, float]] - Expected numeric range for each column (e.g., {'month': (1, 12)}).
    - check_duplicates: bool - Whether to check for duplicate rows in the DataFrame (default=True).
    - check_missing: bool - Whether to check for missing values in the DataFrame (default=True).
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
        Checks data types of DataFrame columns against expected types defined in `column_types`.

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate column data types.

        Raises:
        - TypeError if any column's data type does not match the expected type.
        - ValueError if a required column is missing from the DataFrame.
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
        Validates that numeric columns fall within specified minimum and maximum ranges in `value_ranges`.

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate column value ranges.

        Raises:
        - ValueError if any value in a specified column is out of range.
        """
        
        #Iteration along columns
        for column, (min_value, max_value) in self.value_ranges.items():
            #If any values out of range
            if (X[column] < min_value).any() or (X[column] > max_value).any():
                raise ValueError(f'Values of column {column} are out of expected value range {min_value}-{max_value} ')
    
    def _check_non_negative_values(self: TValidator, X: df) -> Exception | None:

        """
        Checks if all values in numeric columns are non-negative.

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for non-negative values.

        Raises:
        - ValueError if any column contains negative values.
        """
        #Iteration along columns
        for column in X.columns:
            #Negative values detection
            if (X[column] < 0).any():
                raise ValueError(f'Column {column} contains negative values')

    def _check_duplicates(self: TValidator, X: df) -> Exception | None:
    
        """
        Detects and raises an error if there are duplicate rows in the DataFrame.

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for duplicates.

        Raises:
        - ValueError if duplicates are found.
        """
        #If duplicated columns are founded
        if X.duplicated().sum() != 0:
            raise ValueError('Duplicated rows are detected')

    def _check_missing(self: TValidator, X: df) -> Exception | None:
        """
        Detects missing values in columns of the DataFrame.

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for missing values.

        Raises:
        - ValueError if any column contains missing values.
        """
        missing_columns = X.columns[X.isna().any()].tolist()
        #If missing columns are founded
        if missing_columns:
            raise ValueError(f'Columns {missing_columns} contain missing values')


    def fit(self: TValidator, X: df) -> TValidator:
        """
        Fits the Validator by setting column types and value ranges if not specified, based on the provided DataFrame.

        Parameters:
        - X: pd.DataFrame - The DataFrame from which to infer column types and value ranges if not predefined.

        Returns:
        - Validator - Returns self after setting inferred properties.
        """
        if self.column_types is None:
            self.column_types = X.dtypes.to_dict()
        if self.value_ranges is None:
            self.value_ranges = {col: (X[col].min(), X[col].max()) for col in X.columns}

        return self
    
    #validation
    def transform(self: TValidator, X: df) -> str:

        """
        Performs validation checks on a DataFrame, ensuring data types, ranges, and constraints are respected.

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate.

        Returns:
        - str - Confirmation message ("Data is valid") if validation passes.
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
def reduce_mem_usage(df: df, verbose: bool=True) -> df:

    """
    Reduces memory usage of a DataFrame by downcasting numeric columns to more efficient types.

    Parameters:
    - df: pd.DataFrame - The DataFrame to optimize.
    - verbose: bool - Whether to print memory usage reduction details (default=True).

    Returns:
    - pd.DataFrame - DataFrame with optimized memory usage.
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
    return df

#Creating df with full range of data
def full_data_creation(df: df, agg_group: list, periods: int) -> df:

    """
    Generates a DataFrame with the full range of specified item and shop combinations for each period.

    Parameters:
    - df: pd.DataFrame - Input DataFrame with existing data.
    - agg_group: list - List of columns to aggregate (e.g., ['date_block_num', 'shop_id', 'item_id']).
    - periods: int - Number of periods to include in the generated DataFrame.

    Returns:
    - pd.DataFrame - DataFrame containing all combinations of items, shops, and periods.
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


def history_features(df: df, agg: list, new_feature: str) -> df:

    """
    Adds a feature counting the number of unique months for which each combination in `agg` has sales data.

    Parameters:
    - df: pd.DataFrame - Original DataFrame with sales data.
    - agg: list - List of columns to group by (e.g., ['shop_id', 'item_id']).
    - new_feature: str - Name of the new feature to add.

    Returns:
    - pd.DataFrame - DataFrame with the additional feature based on historical sales counts.
    """

    group = (df[df.item_cnt_month > 0].groupby(agg)
            ['date_block_num'].nunique().rename(new_feature).reset_index())

    return df.merge(group, on = agg, how = 'left')

def feat_from_agg(df: df, agg: list, new_col: str, aggregation: dict, output_df: df) -> df:

    """
    Aggregates features based on specified columns, aggregation functions, and adds the result as a new feature.

    Parameters:
    - df: pd.DataFrame - DataFrame for group creation
    - agg: list - Columns to group by (e.g., ['shop_id', 'item_id']).
    - new_col: str - Name for the new aggregated feature.
    - aggregation: Dict[str, Union[str, List[str]]] - Aggregation functions to apply on the grouped data
    - output_df: pd.DataFrame where to add feature

    Returns:
    - pd.DataFrame - DataFrame with the new aggregated feature.
    """
    if new_col == 'first_sales_date_block':
        temp = df[df.item_cnt_month > 0]
    else:
        temp = df.copy()
    temp = temp.groupby(agg).agg(aggregation)
    temp.columns = [new_col]
    temp.reset_index(inplace = True)
    output_df = pd.merge(output_df, temp, on=agg, how='left')

    return output_df

#Lags
def lag_features(df: df, col: str, lags: list = [1,2,3]) -> df:

    """
    Adds lagged features to the DataFrame for specified columns over defined lag periods.

    Parameters:
    - df: pd.DataFrame - DataFrame containing features and target.
    - col: str - Column to create lags for.
    - lags: list - List of lag periods to apply.

    Returns:
    - pd.DataFrame - DataFrame with the newly created lagged features.
    """
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

def new_items(df: df, agg: list, new_col: str) -> df:
    """
    Adds a feature tracking average monthly sales for items with specific historical conditions (e.g., item history of 1).

    Parameters:
    - df: pd.DataFrame - Original DataFrame with items and their monthly sales.
    - agg: list - Columns to group by (e.g., ['shop_id', 'item_id']).
    - new_col: str - Name for the new column.

    Returns:
    - pd.DataFrame - DataFrame with the new column based on items' sales history.
    """

    temp = (df.query('item_history == 1')
    .groupby(agg)
    .agg({'item_cnt_month': 'mean'})
    .reset_index()
    .rename(columns = {'item_cnt_month':new_col}))

    return pd.merge(df, temp, on = agg, how = 'left')


#Data validation using TSS 
def tss_cv(df: df, n_splits, model: Union[LinearRegression, XGBRegressor, LGBMRegressor]) -> np.ndarray:
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

    return y_val, y_pred, model

#Plot of differences between predicted and true values
def true_pred_plot(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
    """
    Plots true versus predicted values to assess model performance visually.

    Parameters:
    - y_true: np.ndarray - Array of true target values.
    - y_pred: np.ndarray - Array of predicted target values.

    Returns:
    - Figure - Scatter plot of true vs predicted values.
    """

    plt.figure(figsize = (10,6))

    #Difference
    sns.scatterplot(x = y_true, y = y_pred, color = 'blue', alpha = 0.5, s = 30, edgecolor = 'k')
    #If prediction will be equal to our target
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')

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

#Prediction creating submission file
def prediction(X_test: np.ndarray, model: Union[LinearRegression, XGBRegressor, LGBMRegressor]) -> str:
    """
    Generates and saves a submission file with predictions for the test set.

    Parameters:
    - X_test: np.ndarray - Feature set for prediction.
    - model: Union[LinearRegression, XGBRegressor, LGBMRegressor] - Trained model to use for predictions.

    Returns:
    - str - Confirmation message indicating submission file was created.
    """

    y_pred = np.round(model.predict(X_test),2).clip(0,20)
    submission = pd.DataFrame({'ID': np.arange(len(y_pred)), 'item_cnt_month': y_pred})
    submission.to_csv('submission.csv', index = False)
    return 'Submission file created'

#Check data leakage
def feature_importancy_check(df: df) -> list[tuple[str, float, float]]:

    """
    Evaluates feature importance for lag features in the dataframe by calculating RMSE 
    and feature importance percentage using cross-validation with XGBRegressor.
    
    Parameters:
    - df: pd.DataFrame - The original dataframe containing features and target.
    Returns:
    - List[Tuple[str, float, float]] - A sorted list of tuples with feature name, RMSE, 
                                       and feature importance percentage.
    """

    #Dataframe without lag features
    base_data = df.loc[:, ~df.columns.str.contains('_lag_')]
    lag_columns = list(df.loc[:, df.columns.str.contains('_lag_')].columns)

    results = []
    #RMSE for baseline
    X_train, y_train, X_val, y_val, X_test = data_split(base_data)
    model = XGBRegressor(n_estimators = 75, eval_metric="rmse", early_stopping_rounds = 20)
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train,y_train, eval_set=eval_set, verbose = False)
    #Biggest Feature importance
    biggest_importance = model.feature_importances_.argsort()[-1]
    feature_importance_percent = model.feature_importances_[biggest_importance] * 100
    # Baseline RMSE, and biggest importance percentage
    results.append(('Baseline', model.best_score, feature_importance_percent))


    merge_group = ['date_block_num', 'shop_id', 'item_id']
    #Iteration through all aggregated features
    for col in tqdm(lag_columns, desc="Feature importance evaluation", ncols=75):

        time.sleep(0.1) 
        
        #Copy of dataframe
        test_data = base_data.merge(df.loc[:, merge_group+[col]], on = merge_group, how = 'left')

        X_train, y_train, X_val, y_val, X_test = data_split(test_data)
        model = XGBRegressor(n_estimators = 75, eval_metric="rmse", early_stopping_rounds = 20)
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train,y_train, eval_set=eval_set, verbose = False)
        
        # Extract feature importance for the specific feature
        feature_importance_percent = model.feature_importances_[-1] * 100

        # Append the feature name, RMSE, and importance percentage
        results.append((col, model.best_score, feature_importance_percent))

    # Sort results by RMSE
    results = sorted(results, key=lambda x: x[1])
    for col, rmse, importance in results:
        print(f"Feature: {col}, RMSE: {rmse}, Importance: {importance:.2f}%")

    return results

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





