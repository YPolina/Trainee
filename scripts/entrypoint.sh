#!/bin/bash

# Authenticate with Google Cloud
echo "Authenticating with Google Cloud..."
gcloud auth activate-service-account --key-file=/opt/airflow/key.json
gcloud auth application-default login

# Pull data with DVC
echo "Pulling data with DVC..."
if ! dvc pull; then
  echo "DVC pull failed. Ensure credentials and remote storage are configured."
  exit 1
fi

# Initialize Airflow database
echo "Initializing Airflow..."
airflow db init

# Create the admin user
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email palinayatsko@innowise.com

# Start the Airflow webserver
echo "Starting Airflow webserver..."
exec airflow webserver
