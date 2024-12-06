import pandas as pd
import gcsfs
import yaml
from pathlib import Path
from pandas.core.frame import DataFrame as df

class DataLoader:
    """Handles data loading from Google Cloud Storage"""

    def __init__(self, data_source = "local", config_path="config.yaml"):
        self.fs = gcsfs.GCSFileSystem()
        self.config = self._load_config(config_path)
        self.data_source = data_source

    def _load_config(self, config_path):
        """Load configuration from a YAML file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load(self, file_key: str) -> pd.DataFrame:
        """
        Load data either from GCS or locally based on config.

        Parameters:
        - file_key: str - Key in the config for the dataset path (e.g., "items").

        Returns:
        - pd.DataFrame - Loaded data.
        """
        # Determine source (GCS or local)
        if self.data_source == "gcs":
            gcs_path = self.config["gcs_paths"][file_key]
            with self.fs.open(gcs_path) as f:
                return pd.read_csv(f)
        elif self.data_source == "local":
            local_path = Path(self.config["local_paths"][file_key])
            return pd.read_csv(local_path)
        else:
            raise ValueError("Invalid data source specified in config.yaml")