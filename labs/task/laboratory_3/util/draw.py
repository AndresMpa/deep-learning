import matplotlib.pyplot as plt
import numpy as np

from util.dirs import create_path, create_dir

from config.vars import env_vars


def draw_views(views, conv, timestamp):
    """
    Draws models views for an activation hook

    Args:
        views (N-array): A normalized activation
        conv (str): A reference for the current convolution
        timestamp (str): A time stamp used as an id
    """
    plt.figure(figsize=(12, 4))

    for i in range(64):
        plt.subplot(4, 16, i + 1)
        plt.imshow(views[i, :, :], cmap="viridis")
        plt.axis("off")

    # To save figure as a picture
    results = env_vars.results_path
    create_dir(results)

    file_name = f'{timestamp}_{conv}_activations'
    file_path = create_path(env_vars.results_path, file_name + ".png")
    plt.savefig(file_path)
    plt.clf()


def draw_error(error, timestamp):
    """
    Plot error (Loss) through epochs

    Args:
        error (N-array): Stores the error to plot
        timestamp (str): A time stamp used as an id
    """
    plt.plot(np.arange(1, len(error) + 1, 1), np.array(error))
    plt.xlabel("Epochs")
    plt.ylabel("Loss function")
    plt.title("Loss through epochs")

    # To save figure as a picture
    results = env_vars.results_path
    create_dir(results)

    file_name = f'{timestamp}_loss_functions'
    file_path = create_path(env_vars.results_path, file_name + ".png")
    plt.savefig(file_path)
    plt.clf()
