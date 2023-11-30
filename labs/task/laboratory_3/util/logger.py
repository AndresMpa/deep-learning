from torch.cuda import is_available
from datetime import datetime

import subprocess
import platform
import csv

from util.dirs import get_current_path, check_path, create_dir, clear_dir

from config.vars import env_vars


def add_entry_to_csv(file_path, new_entry):
    """
    Add a new entry to a CSV file.

    Parameters:
        - file_path (str): The path to the CSV file.
        - new_entry (list): A list containing the data for the new entry.
    """
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(new_entry)


def create_log_file(log_file_path):
    """
    Creates a CSV file to handle with logs

    Parameters:
        - log_file_path (str): The path to the CSV file. This path must be
        checked for get_current_path() function
    """
    entry_labels = [
        'ID - Timestamp',
        "Net Architecture",
        'Training method',
        "Batch size",
        "Iterations",
        "Learning rate",
        "Momentum",
        "Log date"
    ]
    add_entry_to_csv(log_file_path, entry_labels)


def create_log_entry(timestamp):
    """
    Creates an entry in a CSV log file

    Parameters:
        - timestamp (str): A id (Though to be a timestamp)
    """
    log_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file_path = get_current_path(env_vars.log_path)

    if (not check_path(log_file_path)):
        create_dir(log_file_path)

    if (not check_path(log_file_path + "/log_file.csv")):
        create_log_file(log_file_path + "/log_file.csv")

    new_entry = [
        timestamp,
        env_vars.net_arch,
        "Using GPU" if is_available() else "Using CPU",
        env_vars.batch_size,
        env_vars.iterations,
        env_vars.learning_rate,
        env_vars.momentum_value,
        log_datetime
    ]

    add_entry_to_csv(log_file_path + "/log_file.csv", new_entry)


def clear_log():
    if (env_vars.autoclear):
        results = get_current_path(env_vars.results_path)
        log = get_current_path(env_vars.log_path)

        clear_dir(results)
        clear_dir(log)


def expected_time(msg):
    """
    Shows a message in OS' shell using either "print" or "fliglet + lolcat"

    Args:
        msg (str): A message to show
    """
    try:
        if platform.system() == "Windows":
            print(msg)
        else:
            subprocess.run(f"figlet {msg} | lolcat", shell=True)
    except FileNotFoundError:
        print(f"{msg}")
    except subprocess.CalledProcessError as error:
        print(f"Unexpected error: {error}")
