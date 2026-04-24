import csv
import os
from datetime import datetime
from config import LOG_FILE_PATH

def init_logger():
    """Creates the CSV file and headers if it doesn't exist."""
    if not os.path.isfile(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Sign_Type', 'Condition_Status'])
            print(f"[INFO] Created new log file at {LOG_FILE_PATH}")

def log_damage(sign_name, condition_text):
    """Appends a new detection record to the CSV file."""
    with open(LOG_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, sign_name, condition_text])
        print(f"💾 [SAVED TO DATABASE] {timestamp} | {sign_name} | {condition_text}")