import torch.nn as nn
from torch import save
from torch import device
import torch.optim as optim
from torch.cuda import is_available

import torchvision.transforms as transforms

from config.vars import env_vars

from architectures.vgg_16 import VGG16
from architectures.vgg_19 import VGG19
from architectures.alex_net import AlexNet
from architectures.resnet import ResNet, ResidualBlock

from util.dirs import create_path, create_dir
from util.logger import expected_time


def get_architecture():
    """
    Return the selected architecture using .env file

    Returns:
        Architecture class to use
    """
    arch = env_vars.net_arch
    if arch == "AlexNet":
        estimation = env_vars.iterations * 0.0776
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return AlexNet()
    elif arch == "VGG16":
        estimation = env_vars.iterations * 0.12011
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return VGG16()
    elif arch == "VGG19":
        estimation = env_vars.iterations * 2.1046
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return VGG19()
    elif arch == "ResNet":
        estimation = env_vars.iterations * 0.0416667
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return ResNet(ResidualBlock, [2, 2, 2, 2])
    else:
        # Handle the case where the architecture is not recognized
        raise ValueError(f"Unsupported architecture: {arch}")


def create_architecture():
    """
    Creates a new model using the given architecture
    using GPU when possible of CPU by default

    Returns:
        architecture (Object): An instance of the architecture and
        device (Object): The device used (GPU or CPU)
    """
    net_arch = get_architecture()

    if net_arch is None:
        # Handle the case where the architecture class is not found
        raise ValueError("Architecture class is None")

    print(f'Using {"cuda: 0" if is_available() else "cpu"} for processing')

    arch_device = device("cuda:0" if is_available() else "cpu")
    architecture = net_arch

    architecture.to(arch_device)

    return architecture, arch_device


def create_transform():
    """
    Creates a transform using some specific pipeline

    Returns:
        A transforms pipeline composed
    """
    arch = env_vars.net_arch
    if arch == "ResNet":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(env_vars.img_size),
            transforms.ToTensor(),
        ])


def get_loss_function():
    """
    Create a new instance of a loss functions using some criteria from
    .env file

    Returns:
        A loss function using .env file vars to define
        on of them
    """
    if (env_vars.lost_criteria == "CrossEntropyLoss"):
        return nn.CrossEntropyLoss()
    elif (env_vars.lost_criteria == "BCELoss"):
        return nn.BCELoss()


def create_optimizer(architecture):
    """
    Creates an optimizer using hyper parameters from
    .env file

    Returns:
        An instance of an optimizer
    """
    return optim.SGD(
        architecture.parameters(),
        lr=env_vars.learning_rate,
        momentum=env_vars.momentum_value)


def save_arch(arch, timestamp):
    """
    Saves a model to handling with naming and paths creation

    Args:
        arch (Instance): Architecture instance or model to save
        timestamp: A time stamp used as id
    """
    net = env_vars.net_arch
    dataset = env_vars.dataset
    iterations = env_vars.iterations

    results = env_vars.results_path
    create_dir(results)

    file_name = f"{timestamp}_arch-{net}_{dataset}_iter-{iterations}"
    file_path = create_path(results, file_name + ".pth")

    print(f"Created file at: {file_path}")

    save(arch.state_dict(), file_path)
