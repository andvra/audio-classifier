import torch.nn as nn

class PrintLayer(nn.Module):
    """ Use this to make print() while running the sequantial net """
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name}: {x.shape}')
        return x

class ReshapeDynamic(nn.Module):
    """ Reshape input """
    def __init__(self, *dim):
        """ Dim can be any shape """
        super(ReshapeDynamic, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view(*self.dim)

class ConvNet(nn.Module):
    def __init__(self, num_classes, processing_device):
        super(ConvNet, self).__init__()
        self.__num_classes = num_classes
        # This is the number of features after all convolutions
        num_features = 32*2*13
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (4,10), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), # Perform ReLU in-place
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, (4,10), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, (4,10), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ReshapeDynamic((-1, 1, num_features)),
            nn.Linear(num_features, num_classes).to(processing_device)
            )

    def forward(self, x):
        x = self.encoder(x)
        # We don't want the dimension with number of classes (1) to show in the output
        x = x.view(-1, self.__num_classes)
        return x