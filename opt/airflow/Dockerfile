FROM apache/airflow:2.8.1

WORKDIR /opt/airflow

COPY . /opt/airflow
COPY ./dags /opt/airflow/dags
COPY ./key.json /opt/airflow/key.json


USER root

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && pip install --no-cache-dir future_sales_prediction_2024


ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/key.json


COPY . /scripts

ENTRYPOINT ["/entrypoint.sh"]
