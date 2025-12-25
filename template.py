import os # for operating system related functionality
from pathlib import Path # for handling file paths
import logging # for logging messages

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')  # Configure logging format time and message

list_of_files = [ 
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "resarch/trials.ipynb",
    "test.py"
    ]

for filepath in list_of_files:
    filepath = Path(filepath)  # Convert string path to Path object
    filedir, filename = os.path.split(filepath)  # Split into directory and filename
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)  # Create directory if it doesn't exist
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):  # Check if file doesn't exist or is empty
        with open(filepath, "w") as f:  # Create an empty file
            pass
        logging.info(f"Creating new file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}, skipping creation.")