from my_package.data_downloader import download_data_from_gcs

def run_post_install():
    """
    Post-installation script to download raw and preprocessed data
    """
    # Define the GCS paths and local directories
    raw_data_path = "gs://my-dvc-bucket_future_prections/raw_data"
    preprocessed_data_path = "gs://my-dvc-bucket_future_prections/preprocessed_data"

    local_raw_dir = "./raw_data"
    local_preprocessed_dir = "./preprocessed_data"

    # Download data from GCS
    print("Downloading raw data...")
    download_data_from_gcs(raw_data_path, local_raw_dir)

    print("Downloading preprocessed data...")
    download_data_from_gcs(preprocessed_data_path, local_preprocessed_dir)

    print("Data download complete!")
