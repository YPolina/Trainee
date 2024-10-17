from sklearn.base import BaseEstimator, TransformerMixin


#Validation
class Validator(BaseEstimator, TransformerMixin):

    
    def __init__(self, column_types = None, value_ranges = None, non_negative_values = None, check_duplicates = True, check_missing = True):
        #column types:dict
        #Expected data type for each column {'shop_id': 'int64'}
        self.column_types = column_types
        #value ranges:dict
        #Expected value range for numeric columns {'month' : (1, 12)}
        self.value_ranges = value_ranges
        #non_negative_values:list
        #List of columns that should be not negative
        self.non_negative_values = non_negative_values
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
    def check_missing(self, X):
        
        missing_columns = X.columns[X.isna().any()].tolist()
        if missing_columns:
            raise ValueError(f'The following columns have missing values: {missing_columns}')
        
    #check duplicates
    def duplicates(self):
        
        if X.duplicated.any():
            raise ValueError('The dataset contains duplicated rows')

    
    #Fitting model
    def fit(self, X):

        if self.column_types is None:
            self.column_types = {col: X[col].dtype for col in X.columns}
        if self.value_ranges is None:
            self.value_ranges = {col: (X.col.min(), X.col.max()) for col in X.select_dtypes(include = [np.number])}
        if self.non_negative_columns is None:
            self.non_negative_columns = [col for col in X.select_dtypes(include = [np.number])]

        return self

    #Validation checks
    def transform(self, X):

        if self.column_types:
            self._check_column_types()

        if self.value_ranges:
            self._check_value_ranges()

        if self.non_negative_values:
            self._check_non_negative_values()

        if self.check_duplicates:
            self._check_duplicates()
        
        if self.check_missing:
            self._check_missing_values()
        
        return X
        
                                               


    

    


