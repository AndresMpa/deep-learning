import matplotlib.pyplot as plt
import numpy as np

from config.vars import env_vars


def draw_views(views, conv, timestamp):
    """
    draw_views Draws models views for an activation hook

    :param views: A normalized activation
    """
    plt.figure(figsize=(12, 4))

    for i in range(64):
        plt.subplot(4, 16, i + 1)
        plt.imshow(views[i, :, :], cmap="viridis")
        plt.axis("off")

    # To save figure as a picture
    file_name = f'./results/{timestamp}_{conv}_activations.png'
    file_path = (env_vars.results_path, file_name + ".pth")
    plt.savefig(file_path)
    plt.clf()


def draw_error(error, timestamp):
    """
    draw_error Plot error (Loss) through epochs

    :param error: A numpy array
    """
    plt.plot(np.arange(1, len(error) + 1, 1), np.array(error))
    plt.xlabel("Epochs")
    plt.ylabel("Loss function")
    plt.title("Loss through epochs")

    # To save figure as a picture
    file_name = f'./results/{timestamp}_loss_functions.png'
    file_path = (env_vars.results_path, file_name + ".pth")
    plt.savefig(file_path)
    plt.clf()
