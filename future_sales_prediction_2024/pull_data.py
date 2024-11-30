import argparse
import os
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Pull DVC data and clean up unnecessary files.")
    parser.add_argument("--repo", type=str, required=True, help="Git repository URL")
    parser.add_argument("--branch", type=str, default="main", help="Branch to pull from (default: main)")
    args = parser.parse_args()

    # Automatically determine the current directory
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, "data_pulled")
    pull_data(target_dir, args.repo, args.branch)


def pull_data(
    target_dir: str,
    repo_url: str = "https://github.com/YPolina/Trainee.git",
    branch: str = "main"
):
    """
    Clone the repository, pull DVC data, save to current directory, and clean up.

    Parameters:
    - repo_url: str - URL of the Git repository containing DVC metadata
    - target_dir: str - Directory where data will be saved
    - branch: str - The branch of the repository to clone (default: main)
    """

    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Derive repository name from URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    try:
        # Clone the repository
        subprocess.run(["git", "clone", "--branch", branch, repo_url], check=True)

        # Switch to the repository directory
        os.chdir(repo_name)

        # Pull DVC data
        subprocess.run(["dvc", "pull"], check=True)

        # Move relevant data to the target directory
        for folder in ["data/preprocessed_data", "data/raw_data"]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    source_path = os.path.join(folder, file)
                    subprocess.run(["mv", source_path, target_dir], check=True)

        print(f"Data has been pulled and saved to {target_dir}")
    
    finally:
        # Move back to the initial directory
        os.chdir("..")

        # Delete the cloned repository to clean up
        if os.path.exists(repo_name):
            shutil.rmtree(repo_name)

        print(f"Clean-up complete. Only {target_dir} remains.")


if __name__ == "__main__":
    main()
