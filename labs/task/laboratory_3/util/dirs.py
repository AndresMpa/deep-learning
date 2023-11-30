import os


def get_current_path(path):
    """
    Gives the current absolute path

    Args:
        path (str): Path to add into root path

    Returns:
        dir_path (str): Absolute directory path
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    dir_path = os.path.join(root_dir, path)
    return dir_path


def check_path(path):
    """
    Checks if a path exist

    Args:
        path (str): Path to check

    Returns:
        True is exist, False if not
    """
    return (os.path.isdir(path))


def create_path(path, filename):
    """
    Creates an absolute path for a file

    Args:
        path (str): A directory path to host the file
        filename (str): A full file name extension included

    Returns:
        file_path (str): A file path to host a file
    """
    absolute_path = get_current_path(path)
    file_path = os.path.join(absolute_path, filename)

    return file_path


def create_dir(path):
    """
    Creates a given path checking is exists of not

    Args:
        path (str): Path to create
    """
    try:
        if (check_path(path)):
            print(f"{path} already exists")
            return
        else:
            dir_path = get_current_path(path)
            os.mkdir(dir_path)

            print(f"Directory {path} created")
    except OSError:
        print(f"[ERROR]: Creating {path}")


def clear_dir(path):
    """
    Delete all files inside a given directory

    Args:
        path (str): Path to directory to clear
    """
    try:
        path = get_current_path(path)
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print("Files deleted successfully.")
    except OSError:
        print(f"[ERROR]: Deleting files at {path}")
