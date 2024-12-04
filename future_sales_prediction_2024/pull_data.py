import argparse
import os
import shutil
import subprocess
import stat


class DVCDataManager:

    def __init__(
        self,
        repo_url="https://github.com/YPolina/Trainee.git",
        branch="DS-4.1",
        target_dir=None,
    ):
        """
        Initialize DVCDataManager with repository details and target director

        Parameters:
        - repo_url: str - URL of the Git repository
        - branch: str - Branch to clone (default: main)
        - target_dir: str - Directory to save data (default: './data_pulled')
        """
        self.repo_url = repo_url
        self.branch = branch
        self.repo_name = repo_url.split("/")[-1].replace(".git", "")
        self.target_dir = target_dir or os.path.join(os.getcwd(), "data_pulled")

    def clone_repo(self):
        """Clone the Git repository and checkout the specified branch"""
        print(f"Cloning repository {self.repo_url} (branch: {self.branch})")
        subprocess.run(
            ["git", "clone", self.repo_url, "--branch", self.branch], check=True
        )
        os.chdir(self.repo_name)

    def pull_dvc_data(self):
        """Pull DVC-tracked data from the remote storage"""
        print("Pulling data with DVC")
        subprocess.run(["dvc", "pull"], check=True)

    def move_data(self):
        """Move required files from DVC folders to the target directory"""
        print("Moving data to target directory")
        for folder in ["data/preprocessed_data", "data/raw_data"]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    source_path = os.path.join(folder, file)
                    shutil.move(source_path, self.target_dir)

    def clean_non_csv_files(self):
        """Remove all non-CSV files from the target directory."""
        print("Removing non-CSV files from target directory")
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                if not file.endswith(".csv"):
                    os.remove(os.path.join(root, file))
        print("Non-CSV files have been removed")

    def clean_up(self):
        """Clean up temporary repository files"""
        os.chdir("..")
        if os.path.exists(self.repo_name):
            print("Cleaning up temporary repository files")
            shutil.rmtree(self.repo_name, onerror=self.handle_permission_error)
        print("Clean-up complete")

    @staticmethod
    def handle_permission_error(func, path, exc_info):
        """Handle PermissionError by changing file permissions and retrying"""
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)

    def pull_data(self):
        """Main method to pull DVC data and manage files"""
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        try:
            self.clone_repo()
            self.pull_dvc_data()
            self.move_data()
            self.clean_non_csv_files()
            print(f"Data has been successfully saved to {self.target_dir}")
        finally:
            self.clean_up()


def main():
    parser = argparse.ArgumentParser(
        description="Pull DVC data and clean up unnecessary files"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="https://github.com/YPolina/Trainee.git",
        help="Git repository URL",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="DS-4.1",
        help="Branch to pull from (default: DS-4.1)",
    )
    args = parser.parse_args()

    dvc_manager = DVCDataManager(repo_url=args.repo, branch=args.branch)
    dvc_manager.pull_data()


if __name__ == "__main__":
    main()
