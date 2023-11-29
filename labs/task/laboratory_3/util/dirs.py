import os


def get_current_path(path):
    """
    get_current_path Gives the current absolute path

    :param path: Path to add into root path
    :return: Absolute directory path
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(root_dir, path)
    return dir_path


def check_path(path):
    """
    check_path Checks if a path exist

    :param path: Path to check
    :return: True is exist, False if not
    """
    return (os.path.isdir(path))


def create_path(path):
    """
    create_path Creates a given path checking
    is exists of not

    :param path: Path to create
    """
    try:
        if (check_path(path)):
            print(f"{path} already exists")
            return
        else:
            dir_path = get_current_path(path)
            os.mkdir(dir_path)

            print(f"Directory {path:%s} created")
    except OSError:
        print(f"[ERROR]: Creating {path}")


def clear_path(path):
    """
    clear_path Delete all files inside a given directory

    :param path: Path to directory to clear
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
