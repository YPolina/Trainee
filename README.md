#### Future Sales Prediction 2024

Future Sales Prediction 2024 is a Python package designed for building robust time-series sales prediction models. The package integrates preprocessing, feature engineering, hyperparameter optimization, and model training workflows, leveraging DVC for data versioning and Google Cloud Storage for seamless data access.



#### Project Status: Completed

## Features

* Data Handling: Tools to preprocess raw datasets and optimize memory usage.
* Feature Engineering: Generate and refine features for predictive modeling.
* Hyperparameter Tuning: Automate parameter optimization with Hyperopt.
* Model Training: Time-series cross-validation and training for regression models.
* Validation: Validate data integrity to ensure quality and consistency.
* Data Versioning: DVC integration for easy data retrieval from Google Cloud.

### Installation
Install the package using pip:

pip install future_sales_prediction_2024

### Usage Guide
* Step 1: Authenticate with Google Cloud
Before fetching data, authenticate with Google Cloud:

Option A: Use Google Cloud SDK: gcloud auth application-default login

Option B: Use a Service Account key file: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

* Step 2: Pull the Data
Step 2: Pull the Data

Option A - locally:
- Use the pull_data.py script to clone the repository, fetch DVC-tracked data, and save it to the current directory:

* pull_data --repo https://github.com/YPolina/Trainee.git --branch DS-4.1

Option B - using online-service(Google Colab, Kaggle and etc.)
* !pull_data --repo https://github.com/YPolina/Trainee.git --branch DS-4.1

This will:

Clone the repository.
Pull datasets tracked via DVC from Google Cloud Storage.
Save datasets in a folder called data_pulled in the current working directory.

* Step 3: Explore the Codebase and Build Models
After fetching the data, you can explore and use the following modules:

____________________________________________
### Modules and Functions
## Data Handling
File: future_sales_prediction_2024/data_handling.py

prepare_full_data(items, categories, train, shops, test) -> pd.DataFrame
Merges raw datasets into a single comprehensive dataset (full_data.csv), available after dvc pull.

reduce_mem_usage(df) -> pd.DataFrame
Optimizes memory usage by converting data types where applicable.

## Feature Engineering
File: future_sales_prediction_2024/feature_extraction.py

Class: FeatureExtractor
Extracts features for predictive modeling.

Initialization Parameters:
full_data: Full dataset containing all columns.
train: Training data for aggregating revenue-based features.
Output:
Returns a processed dataset (full_featured_data.csv), stored in preprocessed_data after dvc pull.

Class: FeatureImportanceLayer
Analyzes feature importance using baseline and tuned models.

Initialization Parameters:

X: Feature matrix.
y: Target vector.
output_dir: Directory for saving feature importance plots.
Key Methods:

fit_baseline_model(): Trains a baseline model for feature importance based on RandomForestRegressor.
plot_baseline_importance(): Visualizes baseline model feature importance.
fit_final_model(): Trains a final model with optimized hyperparameters - model-agnostic.
Parameters: 
- Model (XGBRegressor by default)
- params: Model hyperparameters (Optional)
- use_shap(bool): Use SHAP values if the model doesn't provide native feature importance
plot_final_model_importance(): Visualizes feature importance for the final model.

Output of plot_baseline_importance and plot_final_model_importance: feature_importance_results/baseline_importance.png and feature_importance_results/final_model_importance.png

## Hyperparameter Tuning
File: future_sales_prediction_2024/hyperparameters.py

hyperparameter_tuning(X, y, model_class, param_space, eval_fn, max_evals=50) -> dict
Performs hyperparameter optimization using Hyperopt for models like XGBRegressor or RandomForestRegressor.

Parameters:

X: Feature matrix.
y: Target vector.
model_class: Model class (e.g., XGBRegressor).
param_space: Search space for hyperparameters.
eval_fn: Evaluation function for loss metric.
max_evals: Number of evaluations.
Returns:
Best hyperparameters as a dictionary.

## Model Training
File: future_sales_prediction_2024/model_training.py

tss_cv(df, n_splits, model, true_pred_plot=True)
Performs time-series cross-validation and calculates RMSE.
Returns Mean RMSE for all splits

df: DataFrame with features and target variable.
n_splits: Number of cross-validation splits.
model: Regression model (e.g., XGBRegressor).
data_split(df) -> Tuple[np.ndarray, ...]
Splits the data into training, validation, and test sets.

train_predict(X, y, X_test, model_, model_params=None) -> np.ndarray
Trains the model with provided features and predicts outcomes.

## Validation
File: future_sales_prediction_2024/validation.py

Class: Validator
Ensures data quality by checking types, ranges, duplicates, and missing values.

Initialization Parameters:

column_types: Expected column data types (e.g., {'shop_id': 'int64'}).
value_ranges: Numeric range for each column (e.g., {'month': (1, 12)}).
check_duplicates: Whether to check for duplicate rows.
check_missing: Whether to check for missing values.
Method: transform(X)
Validates a DataFrame and returns a confirmation message if successful.

### Conclusion:
This package is a modular and flexible solution for streamlining data science workflows. It provides data scientists and ML engineers with reusable tools to focus on solving domain-specific problems.


