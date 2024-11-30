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
python pull_data.py --target data --repo https://github.com/YPolina/Trainee/future_sales_prediction_2024.git --branch DS-4.1
This command:

- Configures DVC to use the correct Google Cloud bucket.
- Fetches all required datasets into the local environment.

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

fit_baseline_model(): Trains a baseline model for feature importance.
plot_baseline_importance(): Visualizes baseline model feature importance.
fit_final_model(): Trains a final model with optimized hyperparameters.
plot_final_model_importance(): Visualizes feature importance for the final model.

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

## [0.1.1] - 2024-11-25
### Added
- Changes in loader function: upload files using filenames.

## [0.2.1] - 2024-11-26
- Added support for Google Cloud Storage.
- Improved deployment pipeline.
- Bug fixes and performance improvements.

## [0.2.2] - 2024-11-27
- Bug fixes.

## [0.2.3] - 2024-11-28
- Enhanced Explainability and Error Analysis
    Users can now save plots generated by the Explainability and ErrorAnalysis classes to files.
    The directory and filenames are customizable, and plots are automatically overwritten if files with the same name already exist.
- Customizable Hyperparameter Tuning
Users can now fully customize the hyperparameter tuning process:
    Define the search space for hyperparameters.
    Specify the optimization algorithm and objective function.
    Tailor the evaluation process to their needs.
- FeatureImportanceLayer Enhancements
    Plots for baseline and final model feature importances can now be saved directly to disk.
    Customizable output directory (output_dir) and file names.
    Plots overwrite existing files with the same name.

## [0.2.4] - 2024-11-29
- Bug fixes.

## [1.2.4] - 2024-11-29
- Cloud Storage Integration
- The data_handling.py and feature_extraction.py scripts now support loading .csv files from GCS paths. Outputs are saved to a user-specified GCS directory via the --outdir parameter.

## [1.2.5, 1.2.6] - 2024-11-29
- Bug fixes.

## [2.2.6] - 2024-11-29
- Automatically fetches raw and preprocessed data from Google Cloud Storage after installation.

## [2.2.7, 2.2.8] - 2024-11-29
- Bug fixes.

## [2.2.9] - 2024-11-30

- The new pull_data.py module clones the repository containing .dvc metadata, fetches data from remote storage (Google Cloud), and saves it to a user-specified target directory.
- The repository URL is fixed to simplify user interaction.
- Improved Workflow:
Users no longer need to manually set up DVC or manage .dvc files.
Everything is handled automatically via a single command.
Fixed Repository URL:
The repository is pre-configured as https://github.com/YPolina/Trainee/future_sales_prediction_2024.git.

## [2.2.10] - 2024-11-30
- Bug fixes.



