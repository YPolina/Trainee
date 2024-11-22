import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import TypeVar
from pandas.core.frame import DataFrame as df

# To use in annotation for the self parameter of class Validator
TValidator = TypeVar("TValidator", bound="Validator")

# Class for validation
class Validator(BaseEstimator, TransformerMixin):
    """
    Initializes Validator with specified column types, value ranges, and options for checking duplicates and missing values

    Parameters:
    - column_types: Dict[str, str] - Expected data types for each column (e.g., {'shop_id': 'int64'})
    - value_ranges: Dict[str, Tuple[float, float]] - Expected numeric range for each column (e.g., {'month': (1, 12)})
    - check_duplicates: bool - Whether to check for duplicate rows in the DataFrame (default=True)
    - check_missing: bool - Whether to check for missing values in the DataFrame (default=True)
    """

    def __init__(
        self: TValidator,
        column_types: dict,
        value_ranges: dict,
        check_duplicates: bool = True,
        check_missing: bool = True,
    ) -> None:

        # Expected data type for each column {'shop_id': 'int64'}
        self.column_types: dict = column_types
        # Expected value range for numeric columns {'month' : (1, 12)}
        self.value_ranges: dict = value_ranges
        # Whether to check duplicates in data
        self.check_duplicates: bool = check_duplicates
        # Whether to check missing values in data
        self.check_missing: bool = check_missing

    # Column types
    def _check_dtypes(self: TValidator, X: df) -> Exception | None:
        """
        Checks data types of DataFrame columns against expected types defined in column_types

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate column data types

        Raises:
        - TypeError if any column's data type does not match the expected type
        - ValueError if a required column is missing from the DataFrame
        """

        # Iteration through all columns
        for column, expected_column_type in self.column_types.items():
            # If we have column in data
            if column in X.columns:
                # Check the equality of dtypes
                if not pd.api.types.is_dtype_equal(
                    X[column].dtype, np.dtype(expected_column_type)
                ):
                    raise TypeError(
                        f"For column {column} expected {expected_column_type} dtype"
                    )
            # If do not have expected column
            else:
                raise ValueError(f"Expected {column} not found")

    def _check_value_ranges(self: TValidator, X: df) -> Exception | None:
        """
        Validates that numeric columns fall within specified minimum and maximum ranges in value_ranges

        Parameters:
        - X: pd.DataFrame - The DataFrame to validate column value ranges

        Raises:
        - ValueError if any value in a specified column is out of range
        """

        # Iteration along columns
        for column, (min_value, max_value) in self.value_ranges.items():
            # If any values out of range
            if (X[column] < min_value).any() or (X[column] > max_value).any():
                raise ValueError(
                    f"Values of column {column} are out of expected value range {min_value}-{max_value} "
                )

    def _check_non_negative_values(self: TValidator, X: df) -> Exception | None:
        """
        Checks if all values in numeric columns are non-negative

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for non-negative values

        Raises:
        - ValueError if any column contains negative values
        """
        # Iteration along columns
        for column in X.columns:
            # Negative values detection
            if (X[column] < 0).any():
                raise ValueError(f"Column {column} contains negative values")

    def _check_duplicates(self: TValidator, X: df) -> Exception | None:
        """
        Detects and raises an error if there are duplicate rows in the DataFrame

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for duplicates

        Raises:
        - ValueError if duplicates are found
        """
        # If duplicated columns are founded
        if X.duplicated().sum() != 0:
            raise ValueError("Duplicated rows are detected")

    def _check_missing(self: TValidator, X: df) -> Exception | None:
        """
        Detects missing values in columns of the DataFrame

        Parameters:
        - X: pd.DataFrame - The DataFrame to check for missing values

        Raises:
        - ValueError if any column contains missing values
        """
        missing_columns = X.columns[X.isna().any()].tolist()
        # If missing columns are founded
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} contain missing values")

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

    # validation
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

        return f"Data is valid"