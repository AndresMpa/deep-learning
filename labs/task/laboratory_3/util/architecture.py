import torch.nn as nn
from torch import save
from torch import device
import torch.optim as optim
from torch.cuda import is_available

import torchvision.transforms as transforms

from config.vars import env_vars

from architectures.alex_net import AlexNet
from architectures.vgg_16 import VGG16
from architectures.vgg_19 import VGG19


def get_architecture():
    """
    get_architecture Return the selected architecture using .env file

    :return: Architecture class to use
    """
    arch = env_vars.net_arch
    if arch == "AlexNet":
        return AlexNet()
    elif arch == "VGG16":
        return VGG16()
    elif arch == "VGG19":
        return VGG19()
    else:
        # Handle the case where the architecture is not recognized
        raise ValueError(f"Unsupported architecture: {arch}")


def create_architecture():
    """
    create_architecture Creates a new model using the given architecture
    using GPU when possible of CPU by default

    :return: architecture which is an instance of the architecture and
    device which is the device used (GPU or CPU)
    """
    net_arch = get_architecture()

    if net_arch is None:
        # Handle the case where the architecture class is not found
        raise ValueError("Architecture class is None")

    arch_device = device("cuda:0" if is_available() else "cpu")
    architecture = net_arch

    architecture.to(arch_device)

    return architecture, arch_device


def create_transform():
    """
    create_transform Creates a transform using some specific pipeline

    :return: A transforms pipeline composed
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(env_vars.img_size),
        transforms.ToTensor(),
    ])


def get_loss_function():
    """
    get_loss_function Returns a loss function using .env file vars to define
    on of them
    """
    if (env_vars.lost_criteria == "CrossEntropyLoss"):
        return nn.CrossEntropyLoss()
    elif (env_vars.lost_criteria == "BCELoss"):
        return nn.BCELoss()


def create_optimizer(architecture):
    """
    create_optimizer Creates an optimizer using hyper parameters from
    .env file

    :return: An optimizer
    """
    return optim.SGD(
        architecture.parameters(),
        lr=env_vars.learning_rate,
        momentum=env_vars.momentum_value)


def save_arch(arch, timestamp):
    """
    save_arch Saves a model to handling with naming and paths creation

    :param arch: Architecture instance or model to save
    """
    net = env_vars.net_net
    dataset = env_vars.dataset
    iterations = env_vars.iterations

    file_name = f"{timestamp}_arch-{net}_{dataset}_iter-{iterations}"
    file_path = (env_vars.results_path, file_name + ".pth")
    save(arch.state_dict(), file_path)
