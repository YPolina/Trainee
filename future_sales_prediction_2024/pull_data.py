import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Pull DVC data from remote storage.")
    parser.add_argument("--repo", type=str, required=True, help="Git repository URL")
    parser.add_argument("--target", type=str, required=True, help="Target directory")
    parser.add_argument("--branch", type=str, default="main", help="Branch to pull from (default: main)")
    args = parser.parse_args()

    pull_data(args.target, args.repo, args.branch)


def pull_data(
    target_dir: str,
    repo_url: str = "https://github.com/YPolina/Trainee/future_sales_prediction_2024.git",
    branch: str = "DS-4.1"
):
    """
    Clone the repository, pull DVC data, and create the target directory

    Parameters:
    - repo_url: str - URL of the Git repository containing DVC metadata
    - target_dir: str - Directory where data will be saved
    - branch: str - The branch of the repository to clone (default: DS-4.1)
    """

    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Derive repository name from URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    # Clone the repository if not already cloned
    if not os.path.exists(repo_name):
        subprocess.run(["git", "clone", repo_url], check=True)

    # Switch to the target branch
    os.chdir(repo_name)
    subprocess.run(["git", "checkout", branch], check=True)

    # Pull DVC data
    subprocess.run(["dvc", "pull"], check=True)

     # Move files from DVC data folder to the target directory
    for folder in ["data/preprocessed_data", "data/raw_data"]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                source_path = os.path.join(folder, file)
                subprocess.run(["mv", source_path, target_dir], check=True)

    print(f"Data has been pulled from branch '{branch}' and saved to {target_dir}")
