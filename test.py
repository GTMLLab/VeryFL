from server.aggregation_alg.fedavg import fedavgAggregator
from model.modelFactory import modelFactory
from client.clients import BaseClient
from client.trainer.fedproxTrainer import fedproxTrainer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from chainfl.interact import chain_proxy

global_args = {
    'client_num': 10,
    'model': 'simpleCNN',
    'dataset': 'fashionMinist',
    'class_num':10,
    'data_folder':'./data',
    'communication_round':200
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
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        #server也类似
        self.server = fedavgAggregator()
        self.client_pool = []
        
        #model就是个工厂类
        self.model = modelFactory().get_model(model=self.global_args.get('model'),class_num=self.global_args.get('class_num'))
        
        #同样类似是client端的训练
        self.trainer = fedproxTrainer
    def construct_dataloader(self):
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataset = DataLoader(dataset=self.train_dataset,batch_size=batch_size,shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset,batch_size=batch_size,shuffle=True)
    
    def construct_client(self):
        for i in range(self.global_args['client_num']):
            client_id = chain_proxy.client_regist()
            new_client = BaseClient(client_id,self.test_dataloader,self.model,self.trainer,1,train_args)
            self.client_pool.append(new_client) 
    def run(self):
        self.construct_dataloader()
        self.construct_client()
        for i in range(self.global_args['communication_round']):
            for client in self.client_pool:
                print(client.train())
            self.server.receive_upload(self.client_pool)
            global_model = self.server.aggregate()
            for client in self.client_pool:
                client.load_state_dict(global_model)

if __name__=="__main__":
    train_dataset = datasets.FashionMNIST(global_args.get('data_folder'),
                                        train=True,
                                        download=True,
                                        transform=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),]))
    test_dataset = datasets.FashionMNIST(global_args.get('data_folder'), 
                                                    train=True, 
                                                    transform=transforms.Compose([transforms.ToTensor()]))
    classification_task = Task(global_args=global_args,
                               train_args=train_args,
                               train_dataset=train_dataset,
                               test_dataset=test_dataset)
    classification_task.run()


