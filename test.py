from server.aggregation_alg.fedavg import fedavgAggregator
from model.modelFactory import modelFactory
from client.clients import BaseClient
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from chainfl.interact import chainProxy 
global_args = {
    'client_num': 10,
    'model': 'resnet',
    'dataset': 'cifar10',
    'class_num':10,
    'data_folder':'./data'
}

train_args = {
    'device': 'cuda',
    'lr':1e-4,
    'weight_decay':1e-5,  
}
class Task:
    def __init__(self,global_args,train_args,train_dataset,test_dataset):
        self.global_args = global_args
        self.train_args = train_args
        
        #等benchmark确定之后，dataset的获取通过工厂类进行。TODO
        self.train_dataset = train_args
        self.test_dataset = test_dataset
        
        #server也类似
        self.server = fedavgAggregator()
        self.client_pool = []
        
        #model就是个工厂类
        self.model = modelFactory().get_model(model=global_args.get('model'),class_num=global_args.get('class_num'))
        
    def construct_dataloader(self):
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataset = DataLoader(dataset=self.train_dataset,batch_size=batch_size,shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset,batch_size=batch_size,shuffle=True)
    
    def construct_client(self):
        
    def run():
        


# train_dataset = datasets.FashionMNIST(global_args.get('data_folder'),
#                                       train=True,
#                                       download=False,
#                                       transform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),]))
# test_dataset = datasets.CIFAR10(global_args.get('data_folder'), 
#                                                 train=False, 
#                                                 transform=transforms.Compose([transforms.ToTensor()]))
# train_dataloader = DataLoader(train_dataset,batch_size = 64,shuffle=True

server = fedavgAggregator()

model = modelFactory().get_model(model="resnet",class_num=10)

clientpool = []
for _ in range(global_args['client_num']):
    clientpool.append()



