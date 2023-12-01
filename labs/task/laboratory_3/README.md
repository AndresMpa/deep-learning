# Project usage

## What is this?

You're seeing a computer vision test field, under the presented architecture, you can easily
try some architecture such as VGGx16 VGGx19 and AlexNet

### Available dataset

Data sets can be defined inside the .env file in the $DATASET$ variable, available dataset
work for image detection or segmentation from PyTorch documentation specification follow this
[link](https://pytorch.org/vision/stable/datasets.html#image-detection-or-segmentation) to see
other options

| Keyword    | Size | Dataset                                                     |
| ---------- | ---- | ----------------------------------------------------------- |
| "CelebA"   | 200K | [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| "CIFAR10"  | 60K  | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)        |
| "CIFAR100" | 60K  | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)        |

> Note: The difference between CIFAR10 and CIFAR100 is the amount of classes, CIFAR10 contains 10
> while CIFAR100 contains 100 see "The CIFAR-100 dataset" specifications

## To run virtual environments

#### Windows

```bath
python -m venv env
env\Scripts\activate
pip3 install numpy python-dotenv matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Linux

```bash
python -m venv env
source env/bin/activate
pip3 install python-dotenv matplotlib numpy torch torchvision torchaudio
```
