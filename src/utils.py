"""
Utility functions for logging, directory management, and artifact serialization.
"""

import os
import json
import logging
import joblib
from typing import Any, Dict, Optional

def setup_logger(name: str = "SentimentAnalysis", log_level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance with console formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def ensure_directories(dirs: list[str]) -> None:
    """Ensures that specified directory paths exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.debug(f"Ensured directory exists: {d}")

def save_artifact(obj: Any, filepath: str) -> str:
    """Serializes and saves a Python object to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    logger.info(f"Saved artifact to: {filepath}")
    return filepath

def load_artifact(filepath: str) -> Any:
    """Loads a serialized Python object from disk using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at: {filepath}")
    obj = joblib.load(filepath)
    logger.info(f"Loaded artifact from: {filepath}")
    return obj

def save_json(data: Dict[str, Any], filepath: str) -> str:
    """Saves a dictionary as a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON metadata to: {filepath}")
    return filepath

def load_json(filepath: str) -> Dict[str, Any]:
    """Loads a JSON file as a dictionary."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found at: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON metadata from: {filepath}")
    return data
