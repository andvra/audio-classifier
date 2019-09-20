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
        # Expected input shape: 64x259
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 2, stride=2), # Result: 32 channels of size 32x129
            nn.BatchNorm2d(32),
            nn.ReLU(), # Perform ReLU in-place
            nn.Conv2d(32, 64, 2, stride=2), # Result: 64 channels of size 16x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, stride=2), # Result: 128 channels of size 8x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ReshapeDynamic((-1, 1, 128*8*32)),
            nn.Linear(128*8*32, num_classes).to(processing_device),
            #PrintLayer('Before softmax'),
            # Since we use Cross-entropy loss, we shouldn't use an activation function here
            #nn.Softmax(dim=2)
            )

    def forward(self, x):
        x = self.encoder(x)
        # We don't want the dimension with number of classes (1) to show in
        #   the output
        x = x.view(-1, self.__num_classes)
        return x