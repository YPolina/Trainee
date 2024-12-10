#!/bin/bash

set -e  # Exit on errors

# Authenticate with Google Cloud
echo "Authenticating with Google Cloud using service account key..."
export GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/key.json
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Navigate to the repository directory
echo "Navigating to the repository directory..."
cd /repository || exit 1

# Pull data with DVC
echo "Pulling data with DVC..."
if ! dvc pull; then
  echo "DVC pull failed. Ensure credentials and remote storage are configured."
  exit 1
fi

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db migrate

# Create an admin user
echo "Creating Airflow admin user..."
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email palinayatsko@innowise.com

# Start the Airflow webserver
echo "Starting Airflow webserver..."
exec airflow webserver
