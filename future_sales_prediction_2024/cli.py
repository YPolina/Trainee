import subprocess
import os

def pull_data():
    """
    dvc pull to upload data from Google Cloud.
    """
    try:
        #DVC settings check
        subprocess.run(["dvc", "remote", "add", "-d", "myremote", "gs://my-dvc-bucket_future_prections", "-f"], check=True)
        
        #dvc pull
        subprocess.run(["dvc", "pull"], check=True)
        print("Данные успешно загружены!")
    except subprocess.CalledProcessError as e:
        print(f"Произошла ошибка при выполнении DVC: {e}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLI для future_sales_prediction_2024")
    parser.add_argument(
        "command",
        choices=["pull-data"],
        help="Available commands: pull-data"
    )
    args = parser.parse_args()

    if args.command == "pull-data":
        pull_data()
