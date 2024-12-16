from airflow import DAG
from airflow.decorators import task, task_group
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator, BranchPythonOperator
from airflow.hooks.base import BaseHook
from airflow.triggers.base import BaseTrigger
from datetime import datetime
import os
import json
import pickle
import pandas as pd
from future_sales_prediction_2024 import (
    MemoryReducer,
    DataLoader,
    MainPipeline,
    Validator,
    Trainer,
    FeatureImportanceLayer,
    HyperparameterTuner,
    Explainability,
    ErrorAnalysis,
)


# Initialize Data Handling class
pipeline = MainPipeline(config_path='/config.yaml')
config = pipeline.config


class DVCHook:
    """
    Custom Hook for DVC operations
    """
    def run_dvc_pull(self, repo_path: str = '/repository'):
        os.chdir(repo_path)
        result = os.system('dvc pull')
        if result != 0:
            raise Exception("DVC pull failed. Ensure credentials and remote storage are configured.")
        print("DVC pull successful.")

def dvc_pull():
    DVCHook().run_dvc_pull('/repository')

def data_prepare(**kwargs):
    """
    Data Handling
    """
    pipeline.run()


def validate_data(**kwargs):
    ti = kwargs['ti']
    print("Data Validation")
    full_featured_data = pipeline.loader.load('full_featured_data')
    MemoryReducer().reduce(full_featured_data)
    column_types = config["validation"]["column_types"]
    values_ranges = config["validation"]["values_ranges"]
    negative_columns = config["validation"]["negative_columns"]
    check_duplicates = config["validation"]["check_duplicates"]
    check_missing = config["validation"]["check_missing"]

    validator = Validator(
        column_types,
        values_ranges,
        negative_columns,
        check_duplicates,
        check_missing,
    )
    validator.transform(full_featured_data)

    full_featured_data.to_parquet(config['intermediates_parquet']['full_featured_data'], index = False)
    del full_featured_data
    print("Data validation completed")


def data_split(**kwargs):
    print("Data Split")
    
    full_featured_data = pd.read_parquet(config['intermediates_parquet']['full_featured_data'])
    trainer = Trainer()

    X_train, y_train, X_test = trainer.split_data(full_featured_data)


    X_train.to_parquet(config['intermediates_parquet']['X_train'], index=False)
    y_train.to_frame().to_parquet(config['intermediates_parquet']['y_train'], index=False)
    X_test.to_parquet(config['intermediates_parquet']['X_test'], index=False)



def hyperparameter_tuning(**kwargs):
    ti = kwargs['ti']

    print("Hyperparameter tuning")
    tuner = HyperparameterTuner(config_path='/config.yaml')

    X_train = pd.read_parquet(config['intermediates_parquet']['X_train'])
    y_train = pd.read_parquet(config['intermediates_parquet']['y_train'])

    _, model_name = tuner.tune(X_train, y_train, model_name = 'XGBRegressor')

    ti.xcom_push(key="model_name", value=model_name)


def train_model(**kwargs):
    ti = kwargs['ti']
    
    model_name = ti.xcom_pull(task_ids='hyperparameter_tuning', key='model_name')


    print("Model training")

    X_train = pd.read_parquet(config['intermediates_parquet']['X_train'])
    y_train = pd.read_parquet(config['intermediates_parquet']['y_train'])
    X_test = pd.read_parquet(config['intermediates_parquet']['X_test'])

    best_params_path = os.path.join(config['artifacts']['params'], 'best_params.json')

    with open(best_params_path, 'r') as file:
        best_params = json.load(file)

    trainer = Trainer()
    y_pred, model, rmse = trainer.train_predict(
        X_train, y_train, X_test, model_name, best_params
    )
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {model_name}")
    print(f"RMSE: {rmse:.4f}")


def feature_importance_layer(**kwargs):

    X_train = pd.read_parquet(config['intermediates_parquet']['X_train'])
    y_train = pd.read_parquet(config['intermediates_parquet']['y_train'])

    best_params_path = os.path.join(config['artifacts']['params'], 'best_params.json')

    with open(best_params_path, 'r') as file:
        best_params = json.load(file)

    print("Feature importance calculations with Baseline Model")

    importance_layer = FeatureImportanceLayer(X_train, y_train)
    importance_layer.fit_baseline_model()
    importance_layer.plot_baseline_importance()
    importance_layer.fit_final_model(params=best_params)
    importance_layer.plot_final_importance()

def explainability_layer(**kwargs):

    print("Explainability layer of trained model")

    model_path = os.path.join(config['artifacts']['models'], 'trained_model.pkl')

    with open(model_path, 'r') as file:
        model = json.load(file)
    expl_layer = Explainability(model=model, X=X_test)
    expl_layer.explaine_instance()
    expl_layer.global_feature_importance()
    for feature in X_test.columns:
        expl_layer.feature_dependence(feature)


def error_analysis_layer(**kwargs):

    X_train = pd.read_parquet(config['intermediates_parquet']['X_train'])
    y_train = pd.read_parquet(config['intermediates_parquet']['y_train'])

    print("Error Analysis layer")

    y_train = y_train.iloc[:, 0]

    error_layer = ErrorAnalysis(X_train, y_train)
    error_layer.train_predict()
    error_layer.model_drawbacks()
    error_layer.large_target_error()
    error_layer.influence_on_error_rate()


# Define the DAG
default_args = {
    'owner': 'Polina Yatsko',
    'start_date': datetime(2023, 12, 6),
    'retries': 1,
}

with DAG(
    'model_training_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    # Authenticate with Google Cloud
    authenticate_task = BashOperator(
        task_id='authenticate_gcloud',
        bash_command=''' 
        echo "Authenticating with Google Cloud using service account key..."
        export GOOGLE_APPLICATION_CREDENTIALS=key.json
        gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
        '''
    )


    # DVC Pull Task
    dvc_pull_task = PythonOperator(
        task_id='dvc_pull',
        python_callable=dvc_pull,
        provide_context=True,
    )

    @task_group(group_id='gcs_dvc')
    def gcs_dvc():
        gcs = EmptyOperator(task_id='authenticate_gcloud')
        dvc = EmptyOperator(task_id='dvc_pull_task')

        gcs >> dvc

    #Validation task
    data_prepare_task = PythonOperator(
        task_id='data_prepare',
        python_callable=data_prepare,
        provide_context=True,
        trigger_rule='none_failed'
    )


    def check_data_exists():
        data_path = '/data/raw_data'
        if not any(file.endswith('.csv') for file in os.listdir(data_path)):
            print("No data found. Pulling data with DVC...")
            return 'gcs_dvc'  
        print("Data already exists. Skipping Google authentication and DVC pull.")
        return 'data_prepare'


    check_data_presence = BranchPythonOperator(
        task_id='check_data_presence',
        python_callable=check_data_exists,
    )

    #Validation task
    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )


    data_split_task = PythonOperator(
        task_id='data_split',
        python_callable=data_split,
        provide_context=True,
    )


    hyperparameter_tuning_task = PythonOperator(
        task_id='hyperparameter_tuning',
        python_callable=hyperparameter_tuning,
        provide_context=True,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
        trigger_rule='none_failed'
    )

    def hyperparams_check():
        params_path = './artifacts/params/best_params.json'
        if os.path.exists(params_path):
            print("Hyperparams are already tunned. Start final model training")
            return 'train_model'
        print("Hyperparams is not found. Start Hyperparameter tunning with hyperopt")
        return 'hyperparameter_tuning'

    check_hyperparams_tuning = BranchPythonOperator(
        task_id='hyperparams_check',
        python_callable=hyperparams_check,
        dag = dag
    )

    feature_importance_layer_task = PythonOperator(
        task_id='feature_importance_layer',
        python_callable=feature_importance_layer,
        provide_context=True,
    )

    explainability_layer_task = PythonOperator(
        task_id='explainability_layer',
        python_callable=explainability_layer,
        provide_context=True,
    )

    error_analysis_layer_task = PythonOperator(
        task_id='error_analysis_layer',
        python_callable=error_analysis_layer,
        provide_context=True,
    )

    @task_group(group_id='analysis')
    def analysis():
        importance_layer = EmptyOperator(task_id='feature_importance_layer')
        explainability_layer = EmptyOperator(task_id='explainability_layer')
        error_layer = EmptyOperator(task_id='error_analysis_layer')


        importance_layer >> explainability_layer >> error_layer

    # Define dependencies
    check_data_presence >> gcs_dvc() >> data_prepare_task >> validate_data_task >> data_split_task >> check_hyperparams_tuning
    check_hyperparams_tuning >> hyperparameter_tuning_task >> train_model_task
    train_model_task >> analysis()