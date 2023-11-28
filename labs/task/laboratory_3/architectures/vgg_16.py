import torch.nn as nn

# Definición del modelo

class VGG16(nn.Module):
  def __init__(self, num_classes = 1000):
    super(VGG16, self).__init__()
    self.characteristic = nn.Sequential(
        # Bloque 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Bloque 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Bloque 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Bloque 4
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Bloque 5
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Bloque 6
    self.flatter = nn.Sequential(
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten()
    )

    self.classificator_nn = nn.Sequential(
        nn.Dropout(),
        nn.Linear(7 * 7 * 512, 4096),                       # FC - 1
        nn.ReLU(inplace = True),
        nn.Linear(4096, 4096),                              # FC - 2
        nn.ReLU(inplace = True),
        nn.Linear(4096, num_classes),                       # FC - 3
        nn.ReLU(inplace = True),
        nn.Softmax(dim = 1)                                 # Output
    )

  def forward(self, x_data):
    value_tracker = self.characteristic(x_data)
    value_tracker = self.flatter(value_tracker)
    value_tracker = value_tracker.view(value_tracker.size(0), -1)
    value_tracker = self.classificator_nn(value_tracker)
    return value_tracker