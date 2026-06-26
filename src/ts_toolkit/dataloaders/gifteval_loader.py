# ts_toolkit/data/gifteval_loader.py
import os
from pathlib import Path

import datasets
from dotenv import load_dotenv

from .base_loader import GluonTSBaseDataset

class GiftEvalDatasetLoader(GluonTSBaseDataset):
    """Loads datasets pre-packaged in the GiftEval (Hugging Face) format."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_hf_dataset(self) -> datasets.Dataset:
        """Loads a specific dataset from the GiftEval storage path."""
        load_dotenv()
        storage_path_str = os.getenv(self.storage_env_var)
        if not storage_path_str:
            raise ValueError(f"Environment variable '{self.storage_env_var}' is not set.")
        
        storage_path = Path(storage_path_str)
        disk_path = storage_path / self.dataset_name
        if not disk_path.exists():
            raise FileNotFoundError(f"Dataset not found at '{disk_path}'")
            
        print(f"Loading Hugging Face dataset from: {disk_path}")
        return datasets.load_from_disk(str(disk_path)).with_format("numpy")