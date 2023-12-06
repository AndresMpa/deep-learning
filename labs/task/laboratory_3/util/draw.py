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
    plt.title(f"Loss through epochs")

    # To save figure as a picture
    results = env_vars.results_path
    create_dir(results)

    file_name = f'{timestamp}_loss_functions'
    file_path = create_path(env_vars.results_path, file_name + ".png")
    plt.savefig(file_path)
    plt.clf()

def confusion_matrix(y_true, y_pred, num_classes):
    pass

# On develop
def get_metric(conf_matrix):
    accuracy = (conf_matrix.diagonal().sum()) / conf_matrix.sum()
    precision = conf_matrix.diagonal() / conf_matrix.sum(axis=0)
    recall = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Since dividing by 0 is expected, this line ignores it
    np.seterr(divide='ignore', invalid='ignore')

    print(f"Accuracy: \n{accuracy}")
    print(f"Precision: \n{precision}")
    print(f"Recall: \n{recall}")
    print(f"F1-Score: \n{f1_score}")

# On develop
def draw_confusion_matrix(conf_matrix):
    # Figure size
    plt.figure(figsize=(8, 6))

    # To create a heatmap
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

    # To plot values per cell
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

    # Labels and titles
    plt.xticks(range(len(conf_matrix)), [f'Clase {index + 1}' for index in range(len(conf_matrix))])
    plt.yticks(range(len(conf_matrix)), [f'Clase {index + 1}' for index in range(len(conf_matrix))])
    plt.xlabel('Predicted class')
    plt.ylabel('Expexted class')
    plt.title('Confusion matrix')
    plt.show()