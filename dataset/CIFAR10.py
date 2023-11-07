from torchvision import datasets, transforms
from config.dataset import dataset_file_path

def get_cifar10(train:bool = True):
    return datasets.CIFAR10(root = dataset_file_path,
                                train = train,
                                download = True,
                                transform = transforms.Compose([transforms.RandomCrop(32,padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),]))
    