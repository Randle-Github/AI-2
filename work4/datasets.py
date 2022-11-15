from heads import *

rcParams['font.sans-serif'] = ['simhei']
rcParams['axes.unicode_minus'] = False

cifarTrainingTransforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
])

cifarTraining = torchvision.datasets.CIFAR10("./", train=True, download=True, transform=cifarTrainingTransforms)
cifarTest     = torchvision.datasets.CIFAR10("./", train=False, download=True, transform=T.ToTensor())


mnistTransforms = T.Compose([
    T.ToTensor(),
])

mnistTraining = torchvision.datasets.MNIST("./", train=True, download=True, transform=mnistTransforms)
mnistTest     = torchvision.datasets.MNIST("./", train=False, download=True, transform=mnistTransforms)