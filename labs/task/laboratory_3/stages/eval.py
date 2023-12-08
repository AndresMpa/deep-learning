import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from util.dataset import create_dataset
from util.architecture import use_model, create_device, create_transform

from util.logger import create_log_entry, send_message_to_os


def execute_eval():
    start_time = time.time()

    """
    Model definition
    """
    model, model_name = use_model()
    transform = create_transform(model_name)
    device = create_device()

    print("Using model architecture:")
    print(model)

    """
    Dataset split
    """
    _, testloader, _, testset = create_dataset(transform, True)
    class_names = testset.classes

    """
    Setting eval function
    """
    model.eval()

    total = 0
    correct = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)

            outputs = model(input)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Print batch-level information
            if batch_idx % 10 == 0:
                batch = f"Batch [{batch_idx}/{len(testloader)}]"
                accuracy = f"Accuracy: {correct / total * 100:.2f}%"
                print(f"{batch}: {accuracy}")

    # Print overall accuracy
    accuracy = correct / total
    print(f"Accuracy on the sample dataset: {accuracy * 100:.2f}%")

    """
    Measuring time
    """
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"Took: {elapsed_time:0.2f} minutes")

    timestamp = time.time()

    """
    Metrics
    """
    conf_matrix = np.zeros(
        (len(class_names), len(class_names)), dtype=np.int64)

    for label, prediction in zip(true_labels, predicted_labels):
        conf_matrix[label, prediction] += 1

    """
    Plotting
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=45, ha="right")

    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    plt.ylabel('Classes')
    plt.title(f"{model_name} confusion matrix")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

    '''
    Logs
    '''
    create_log_entry(timestamp, elapsed_time)
    send_message_to_os(
        f"Process ended; took {elapsed_time} minutes",
        f"{model_name}"
    )
