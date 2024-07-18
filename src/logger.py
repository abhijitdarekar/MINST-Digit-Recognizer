import sys
import os
import logging

from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y')}.log"#%H_%M_%S')}.log"
LOG_FOLDER = f"{datetime.now().strftime('%m_%d_%Y')}"

# Create Current Day Log Folder
log_folder_path = os.path.join(os.getcwd(),'logs',LOG_FOLDER)
os.makedirs(log_folder_path,exist_ok=True)

log_file_path = os.path.join(log_folder_path, LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s | %(levelname)s | %(message)s",
    level=logging.DEBUG,
)
