FROM apache/airflow:latest

USER root

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libgomp1 \ 
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && apt-get clean

USER airflow

COPY airflow_requirements.txt /opt/airflow/airflow_requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/airflow_requirements.txt

ENV AIRFLOW_HOME=/opt/airflow