from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import json
from future_sales_prediction_2024 import (
    MemoryReducer,
    MainPipeline,
    DataLoader,
    Validator,
    Trainer,
    FeatureImportanceLayer,
    HyperparameterTuner,
    Explainability,
    ErrorAnalysis,
)

loader = DataLoader()

def load_data(**kwargs):
    """
    Data Load for executing the pipeline.

    Parameters:
    - new_test_data(bool): True - If True, the user will pass their own test set
    - test_file: Path to a file in .csv format if `new_test_data` is True
    """
    config = kwargs['dag_run'].conf

    full_featured_data = loader.load("full_featured_data")
    MemoryReducer().reduce(full_featured_data)

    # Pass the data to the next tasks using XCom
    kwargs['ti'].xcom_push(key='full_featured_data', value=full_featured_data)

    return None



def validate_data(**kwargs):

    # Step 2. Data Validation
    print("Data Validation")
    full_featured_data = kwargs['ti'].xcom_pull(key='full_featured_data')

    config = loader.config
    column_types = config["validation"]["column_types"]
    values_ranges = config["validation"]["values_ranges"]
    negative_columns = config["validation"]['negative_columns']
    check_duplicates=config["validation"]["check_duplicates"]
    check_missing=config["validation"]["check_missing"]

    validator = Validator(
        column_types,
        values_ranges,
        negative_columns,
        check_duplicates,
        check_missing,
    )
    validator.transform(full_featured_data)
    return "Data validation completed."

def data_split(**kwargs):
    print("Data Split")

    full_featured_data = kwargs['ti'].xcom_pull(key='full_featured_data')
    trainer = Trainer()
    X_train, y_train, X_test = trainer.split_data(full_featured_data)
    
    # Pass training and test sets to the next tasks
    kwargs['ti'].xcom_push(key='X_train', value=X_train)
    kwargs['ti'].xcom_push(key='y_train', value=y_train)
    kwargs['ti'].xcom_push(key='X_test', value=X_test)

    return None


def hyperparameter_tunning(**kwargs):
    print("Hyperparameter tunning")
    X_train = kwargs['ti'].xcom_pull(key='X_train')
    y_train = kwargs['ti'].xcom_pull(key='y_train')

    tuner = HyperparameterTuner()
    best_params, model_name = tuner.tune(X_train, y_train)

    # Pass best parameters and model name to the next task
    kwargs['ti'].xcom_push(key='best_params', value=best_params)
    kwargs['ti'].xcom_push(key='model_name', value=model_name)

    return None

def train_model(**kwargs):
    print("Model training")

    X_train = kwargs['ti'].xcom_pull(key='X_train')
    y_train = kwargs['ti'].xcom_pull(key='y_train')
    X_test = kwargs['ti'].xcom_pull(key='X_test')
    model_name = kwargs['ti'].xcom_pull(key='model_name')
    best_params = kwargs['ti'].xcom_pull(key='best_params')

    trainer = Trainer()
    y_pred, model, rmse = trainer.train_predict(
        X_train, y_train, X_test, model_name, best_params
    )
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {model_name}")
    print(f"RMSE: {rmse:.4f}")

    # Pass predictions and model to the next tasks
    kwargs['ti'].xcom_push(key='model', value=model)
    kwargs['ti'].xcom_push(key='y_pred', value=y_pred)

    return None


def feature_importance_layer(**kwargs):

    print("Feature importance calculations with Baseline Model")
    X_train = kwargs['ti'].xcom_pull(key='X_train')
    y_train = kwargs['ti'].xcom_pull(key='y_train')
    best_params = kwargs['ti'].xcom_pull(key='best_params')

    importance_layer = FeatureImportanceLayer(X_train, y_train)
    importance_layer.fit_baseline_model().plot_baseline_importance()

    print(f"Feature importance calculations with {model_name}")
    feature_layer.fit_final_model(params=best_params)
    feature_layer.plot_final_importance()

    return None

def explainability_layer(**kwargs):
    print("Explainability layer of trained model")

    model = kwargs['ti'].xcom_pull(key='model')
    X_test = kwargs['ti'].xcom_pull(key='X_test')

    expl_layer = Explainability(model=model, X=X_test)
    expl_layer.explaine_instance()
    expl_layer.global_feature_importance()
    for feature in X_train.columns:
        expl_layer.feature_dependence(feature)
    return None

def error_analysis_layer(**kwargs):
    print("Error Analysis layer")

    X_train = kwargs['ti'].xcom_pull(key = 'X_train')
    y_train = kwargs['ti'].xcom_pull(key = 'y_train')

    error_layer = ErrorAnalysis(X_train, y_train)
    error_layer.train_predict()
    error_layer.model_drawbacks()
    error_layer.large_target_error()
    error_layer.influence_on_error_rate()

    return None

default_args = {
    'owner': 'Polina Yatsko',
    'start_date': datetime(2023, 12, 6),
    'retries': 1,
}


with DAG('model_training_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True
    )

    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True
    )

    data_split_task = PythonOperator(
        task_id='data_split',
        python_callable=data_split,
        provide_context=True
    )
    hyperparameter_tunning_task = PythonOperator(
        task_id='hyperparameter_tunning',
        python_callable=hyperparameter_tunning,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True
    )

    feature_importance_layer_task = PythonOperator(
        task_id='feature_importance_layer',
        python_callable=feature_importance_layer,
        provide_context=True
    )

    explainability_layer_task = PythonOperator(
        task_id='explainability_layer',
        python_callable=explainability_layer,
        provide_context=True
    )

    error_analysis_layer_task = PythonOperator(
        task_id='error_analysis_layer',
        python_callable=error_analysis_layer,
        provide_context=True
    )

    # Task dependencies
    load_data_task >> validate_data_task >> data_split_task >> hyperparameter_tuning_task >> train_model_task
    train_model_task >> [feature_importance_layer_task, explainability_layer_task, error_analysis_layer_task]