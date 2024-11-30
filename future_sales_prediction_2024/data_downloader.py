import os
import gcsfs

def download_data_from_gcs(bucket_path, local_dir):
    """
    Download files from a GCS bucket to a local directory.

    Parameters:
    - bucket_path: str - Path to the folder in the GCS bucket containing the data
    - local_dir: str - Path to the local directory where data will be downloaded
    """
    fs = gcsfs.GCSFileSystem()

    # List all files in the bucket path
    files = fs.glob(f"{bucket_path}/*")
    os.makedirs(local_dir, exist_ok=True)

    # Download each file
    for file in files:
        file_name = os.path.basename(file)
        local_file_path = os.path.join(local_dir, file_name)
        with fs.open(file, 'rb') as src_file, open(local_file_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
        print(f"Downloaded {file} to {local_file_path}")
