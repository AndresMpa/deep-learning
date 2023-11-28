import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import multiprocessing

import matplotlib.pyplot as plt

import numpy as np

import os

from architectures.vgg_16 import VGG16


if __name__ == '__main__':
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
    # Path de descarga
    data_path = "./data"

    # Descargar y cargar el conjunto de datos CIFAR-10

    # ~5200 datos
    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=os.path.isdir(data_path), transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=os.path.isdir(data_path), transform=transform)

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
                    epochs, j+1, iteration_lost / 200))
                error.append(iteration_lost)
                iteration_lost = 0

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
        plt.subplot(4, 16, i)
        plt.imshow(activation_normalized[i, :, :], cmap="viridis")
        plt.axis("off")
    plt.show()

    # Se visualiza la función de perdida
    plt.plot(np.arange(1, len(np.array(error)) + 1, 1), np.array(error))
    plt.xlabel("Epochs")
    plt.ylabel("Loss function")
    plt.show()

    # Graba un modelo entrenado
    torch.save(architecture.state_dict(), 'architecture_cifar10.pth')
