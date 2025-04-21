import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def ensure_dir_exists(directory):
    """
    Ensure that a directory exists. Create it if it doesn't.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def file_exists(filepath):
    """
    Check if a file exists.

    Args:
        filepath (str): Path to the file.
    
    Returns:
        bool: True if the file exists, False otherwise.
    """
    exists = os.path.isfile(filepath)
    if not exists:
        logging.warning(f"File does not exist: {filepath}")
    return exists

def validate_numeric(value, value_name="Value"):
    """
    Ensure that a value is numeric (int or float).

    Args:
        value: The value to validate.
        value_name (str): Name of the value for error messages.
    
    Raises:
        ValueError: If the value is not numeric.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{value_name} must be a numeric type (int or float). Got: {type(value).__name__}")

def current_timestamp():
    """
    Get the current timestamp in ISO 8601 format.

    Returns:
        str: Current timestamp.
    """
    return datetime.now().isoformat()

class NebulaPyError(Exception):
    """
    Custom exception class for NebulaPy-specific errors.
    """
    pass
