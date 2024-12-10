FROM apache/airflow:2.10.0

WORKDIR /opt/airflow

# Copy files into the container
COPY . /opt/airflow
COPY ./dags /opt/airflow/dags
COPY . /airflow/airflow.cfg
COPY ./key.json /opt/airflow/key.json
COPY requirements.txt /opt/airflow/requirements.txt

USER root

# Install dependencies for Google Cloud SDK and Swagger UI
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-sdk

# Switch to the 'airflow' user
USER airflow

# Install custom Python package and Swagger UI
RUN pip install --no-cache-dir future_sales_prediction_2024 && \
    pip install --no-cache-dir -r /opt/airflow/requirements.txt && \
    pip install apache-airflow-providers-openlineage==1.4.0

# Set environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/key.json

# Copy the entrypoint script
COPY . scripts/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/scripts/entrypoint.sh"]
