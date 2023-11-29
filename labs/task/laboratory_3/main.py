import torch
import torch.nn as nn

import multiprocessing

import matplotlib.pyplot as plt

import numpy as np

import time

from util.architecture import create_architecture
from util.architecture import create_optimizer
from util.architecture import create_transform
from util.dataset import create_dataset

from config.vars import env_vars

if __name__ == '__main__':
    start_time = time.time()

    """
    Threads management
    """
    multiprocessing.freeze_support()

    """
    Architecture definition
    """
    architecture, device = create_architecture()
    transform = create_transform()

    """
    Hiperparametros
    """
    lost_criteria = nn.CrossEntropyLoss()
    optimizator = create_optimizer(architecture)

    """
    Analytics
    """
    error = []

    """
    Dataset
    """
    trainloader, testloader = create_dataset(transform)

    """
    Hooks
    """
    # Función para registrar las activaciones de las capas intermedias
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    def hook(module, input, output):
        global activation
        activation = output.detach()

    print(architecture.characteristic)

    # Registrar ganchos (hooks) en capas intermedias
    architecture.characteristic[3].register_forward_hook(
        get_activation('conv1'))
    architecture.characteristic[6].register_forward_hook(
        get_activation('conv2'))

    '''
    Trainig process
    '''

    for epochs in range(env_vars.iterations):
        iteration_lost = 0
        for j, i in enumerate(trainloader, 0):
            X, Y = i
            X, Y = X.to(device), Y.to(device)

            optimizator.zero_grad()
            output = architecture(X)

            lost = lost_criteria(output, Y)
            lost.backward()

            optimizator.step()
            iteration_lost += lost

            if (j % 200 == 0):
                print(f'[{epochs} {j:5d} {(iteration_lost / 200):.3f}]')
                error.append(iteration_lost)
                iteration_lost = 0

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Took: {elapsed_time / 60.0:0.2f} minutes (Processing on GPU)")

    # Move the model back to the CPU
    architecture.to("cpu")

    '''
    Visualization
    '''
    # Registra el gancho en la capa de interés
    activation = activations["conv1"]

    # Index de la imagen a visualizar
    index = 0

    # Se normaliza la activación (Atenua pixeles para que no tenga mucho brillo)
    activation = activation[index, :, :, :].cpu().numpy()

    activation_normalized = (activation - activation.min()) / \
        (activation.max() - activation.min())

    # Se visualizan las activaciones
    plt.figure(figsize=(12, 4))

    for i in range(64):
        plt.subplot(4, 16, i + 1)
        plt.imshow(activation_normalized[i, :, :], cmap="viridis")
        plt.axis("off")

    # To save figure as a picture
    plt.savefig('./results/activations.png')

    '''
    Loss Function Plot
    '''

    # Move the error to the CPU before plotting
    error_cpu = [e.cpu().item() for e in error]

    # Se visualiza la función de perdida
    plt.plot(np.arange(1, len(error_cpu) + 1, 1), np.array(error_cpu))
    plt.xlabel("Epochs")
    plt.ylabel("Loss function")
    plt.title("Loss through epochs")

    # To save figure as a picture
    plt.savefig('./results/loss_functions.png')

    # Graba un modelo entrenado
    torch.save(architecture.state_dict(), './results/architecture_cifar10.pth')
