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

def load_data(**kwargs):
    """
    Data Load for executing the pipeline.

    Parameters:
    - new_test_data(bool): True - If True, the user will pass their own test set
    - test_file: Path to a file in .csv format if `new_test_data` is True
    """
    conf = kwargs["dag_run"].conf
    test_data = conf.get("test_data", [])
    test_file_path = "./data/raw_data/test_new_data.csv"
    pd.DataFrame(test_data).to_csv(test_file_path, index=False)
    
    loader = DataLoader()
    full_featured_data = loader.load("full_featured_data")
    MemoryReducer().reduce(full_featured_data)

    # Pass the data to the next tasks using XCom
    kwargs['ti'].xcom_push(key='full_featured_data', value=full_featured_data)

    return None


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

    model_name = conf.get("model_name", None)

    tuner = HyperparameterTuner()
    best_params, model_name = tuner.tune(X_train, y_train, model_name=model_name)

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
        X_train, y_train, X_test, model_name, best_params, save_model = False
    )
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {model_name}")
    print(f"RMSE: {rmse:.4f}")

    # Pass predictions and model to the next tasks
    kwargs['ti'].xcom_push(key='model', value=model)
    kwargs['ti'].xcom_push(key='y_pred', value=y_pred)

    return None

def save_predictions(y_pred, test_data_path):
    test_data_df = pd.read_csv(test_data_path)
    y_pred = np.array(y_pred).flatten()
    predictions = [
        {"ID": int(row["ID"]), "prediction": float(pred)}
        for row, pred in zip(test_data_df.to_dict("records"), y_pred)
    ]
    with open("./data/predictions.json", "w") as f:
        json.dump(predictions, f)



default_args = {
    'owner': 'Polina Yatsko',
    'start_date': datetime(2023, 12, 6),
    'retries': 1,
}


with DAG('fastapi_training_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
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

    # Task dependencies
    load_data_task >> data_split_task >> hyperparameter_tunning_task >> train_model_task