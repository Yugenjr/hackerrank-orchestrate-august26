"""
DataLoader Module for Message Notification Router
Handles efficient loading and indexing of all CSV datasets.
"""
import os
import logging
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        # Store as pure Python dicts for absolute max performance (O(1) lookups without pandas Series overhead)
        self.data_dicts: Dict[str, Any] = {}
        
        # Define expected files and their index columns
        self.index_config: Dict[str, Any] = {
            "users.csv": "user_id",
            "groups.csv": "group_id",
            "group_members.csv": ["group_id", "user_id"],
            "business_accounts.csv": "business_id",
            "user_business_history.csv": ["user_id", "business_id"],
            "daily_notification_summary.csv": ["user_id", "date"],
            "messages.csv": "message_id",
            "message_history.csv": "message_id",
            "message_events.csv": ["user_id", "message_id"]
        }

    def load_all(self) -> None:
        """Loads all configured CSV files into memory as nested dictionaries."""
        logger.info(f"Loading datasets from {self.dataset_dir}...")
        for filename, index_col in self.index_config.items():
            # Use os.path.abspath to prevent basic path traversal via dataset_dir
            file_path = os.path.abspath(os.path.join(self.dataset_dir, filename))
            # Verify the file_path is strictly inside dataset_dir
            if not file_path.startswith(os.path.abspath(self.dataset_dir)):
                logger.error(f"Security: Path traversal blocked for {filename}")
                continue
                
            self._load_csv(filename, file_path, index_col)
            
    def _load_csv(self, filename: str, file_path: str, index_col: Any) -> None:
        """Loads a single CSV and converts it to a native dictionary indexed by the primary key."""
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}. Skipping.")
            return
            
        try:
            # Force all IDs to strings to prevent float casting (e.g. 123 -> 123.0)
            dtypes = {}
            if isinstance(index_col, list):
                for col in index_col:
                    dtypes[col] = str
            else:
                dtypes[index_col] = str
                
            df = pd.read_csv(file_path, dtype=dtypes)
            
            # Replace NaNs with None for cleaner dict usage
            df = df.replace({np.nan: None})
            
            if isinstance(index_col, list):
                if all(col in df.columns for col in index_col):
                    df.set_index(index_col, inplace=True)
                else:
                    logger.error(f"Missing expected index columns {index_col} in {filename}")
                    return
            elif index_col in df.columns:
                df.set_index(index_col, inplace=True)
            else:
                logger.error(f"Missing expected index column {index_col} in {filename}")
                return
                
            self.data_dicts[filename] = df.to_dict('index')
            logger.debug(f"Loaded {filename} with {len(self.data_dicts[filename])} records.")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}", exc_info=True)
            raise

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._get_row("users.csv", user_id)

    def get_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        return self._get_row("groups.csv", group_id)

    def get_group_member(self, group_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        return self._get_row("group_members.csv", (group_id, user_id))

    def get_business(self, business_id: str) -> Optional[Dict[str, Any]]:
        return self._get_row("business_accounts.csv", business_id)

    def get_user_business_history(self, user_id: str, business_id: str) -> Optional[Dict[str, Any]]:
        return self._get_row("user_business_history.csv", (user_id, business_id))
        
    def get_daily_summary(self, user_id: str, date: str) -> Optional[Dict[str, Any]]:
        return self._get_row("daily_notification_summary.csv", (user_id, date))

    def get_message_event(self, user_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        return self._get_row("message_events.csv", (user_id, message_id))

    def _get_row(self, filename: str, key: Any) -> Optional[Dict[str, Any]]:
        """Generic method to fetch a row by index key using native python dicts."""
        data_dict = self.data_dicts.get(filename)
        if data_dict is None:
            return None
        return data_dict.get(key)
