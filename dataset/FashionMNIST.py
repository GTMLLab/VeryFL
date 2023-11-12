from torchvision import datasets, transforms
from config.dataset import dataset_file_path

def get_fashionmnist(train:bool = True):
    return datasets.FashionMNIST(root = dataset_file_path,
                                train = train,
                                download = True,
                                transform = transforms.Compose([transforms.RandomCrop(28,padding=4),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor(),]))
    