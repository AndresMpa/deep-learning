import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import multiprocessing

import matplotlib.pyplot as plt

import numpy as np

import time

import os

from architectures.vgg_16 import VGG16


if __name__ == '__main__':
    start_time = time.time()
    multiprocessing.freeze_support()

    # Crear una instancia de VGG16
    architecture = VGG16()

    # Verificar si hay una GPU disponible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Mover el modelo a la GPU si está disponible
    architecture.to(device)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ])

    # Función para registrar las activaciones de las capas intermedias
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    def hook(module, input, output):
        global activation
        activation = output.detach()

    # Registrar ganchos (hooks) en capas intermedias
    architecture.characteristic[3].register_forward_hook(
        get_activation('conv1'))
    architecture.characteristic[6].register_forward_hook(
        get_activation('conv2'))

    '''
    Se trae el dataset
    '''

    # Descargar y cargar el conjunto de datos CIFAR-10
    data_path = "./data"

    # ~5200 datos
    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=(not os.path.isdir(data_path)), transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=(not os.path.isdir(data_path)), transform=transform)

    # ~900 datos
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=8, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True, num_workers=2)

    '''
    Hiperparametros
    '''

    iterations = 1
    learning_rate = 0.01
    momentum_value = 0.8
    lost_criteria = nn.CrossEntropyLoss()
    optimizator = optim.SGD(architecture.parameters(),
                            lr=learning_rate, momentum=momentum_value)

    """
    Analytics
    """

    error = []

    '''
    Trainig process
    '''

    for epochs in range(iterations):
        iteration_lost = 0
        for j, i in enumerate(trainloader, 0):
            X, Y = i
            X, Y = X.to(device), Y.to(device)
            # Se inicializan los gradientes
            optimizator.zero_grad()
            # Se pasa la data por toda la architecture (Fowarding data)
            output = architecture(X)
            # Se calcula la perdida del modelo
            lost = lost_criteria(output, Y)
            # Backward propagation
            lost.backward()
            # Se actualizan los pesos, es decir; se da un paso
            optimizator.step()
            # Para almacenar la perdida en cada epoca
            iteration_lost += lost

            if (j % 200 == 0):
                print('[{} {:5d} {:.3f}]'.format(
                    epochs, j, iteration_lost / 200))
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
