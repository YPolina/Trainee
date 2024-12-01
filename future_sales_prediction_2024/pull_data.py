import argparse
import os
import shutil
import subprocess
import stat


def main():
    parser = argparse.ArgumentParser(
        description="Pull DVC data and clean up unnecessary files."
    )
    parser.add_argument("--repo", type=str, required=True, help="Git repository URL")
    parser.add_argument(
        "--branch", type=str, default="main", help="Branch to pull from (default: main)"
    )
    args = parser.parse_args()

    # Automatically determine the current directory
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, "data_pulled")
    pull_data(target_dir, args.repo, args.branch)


def pull_data(
    target_dir, repo_url="https://github.com/YPolina/Trainee.git", branch="main"
):
    """
    Clone the repository, pull DVC data, save to target directory, and clean up.

    Parameters:
    - target_dir: str - Directory where data will be saved
    - repo_url: str - URL of the Git repository containing DVC metadata
    - branch: str - Branch to clone (default: main)
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    repo_name = repo_url.split("/")[-1].replace(".git", "")

    try:
        # Clone the repository and pull DVC data
        print(f"Cloning repository {repo_url} (branch: {branch})...")
        subprocess.run(["git", "clone", "--branch", branch, repo_url], check=True)
        os.chdir(repo_name)
        print("Pulling data with DVC...")
        subprocess.run(["dvc", "pull"], check=True)

        # Move required files to target directory
        print("Moving data to target directory...")
        for folder in ["data/preprocessed_data", "data/raw_data"]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    source_path = os.path.join(folder, file)
                    shutil.move(source_path, target_dir)

        # Remove all non-CSV files in target directory
        print("Removing non-CSV files from target directory...")
        remove_non_csv_files(target_dir)
        print(f"Data has been successfully saved to {target_dir}")

    finally:
        os.chdir("..")
        if os.path.exists(repo_name):
            print("Cleaning up temporary repository files...")
            shutil.rmtree(repo_name, onerror=handle_permission_error)
        print("Clean-up complete.")



def remove_non_csv_files(target_dir):
    """
    Remove all non-CSV files from the target directory.

    Parameters:
    - target_dir: str - Directory to clean
    """
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if not file.endswith(".csv"):
                os.remove(os.path.join(root, file))
    print("Non-CSV files have been removed.")


def handle_permission_error(func, path, exc_info):
    """
    Handle PermissionError by changing file permissions and retrying.
    """
    if not os.access(path, os.W_OK):
        #User's permission
        os.chmod(path, stat.S_IWUSR)
        #Fuction re-run
        func(path)

if __name__ == "__main__":
    main()
