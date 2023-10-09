from client.trainer.fedproxTrainer import fedproxTrainer
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch


if __name__ =="__main__":
    #-----------Unit test for serverSimulator.py----------------
    #This test contains 
    #1. The functional of the interact with the server(upload and download model)
    #2. The Factory of aggregator for user to choose the fl algorithm (TODO)
    #3. Args passed into the server.
    #4. 
    from server.serverSimulator import serverSimulator
    from server.base.baseAggregator import ServerAggregator 
    import torch.nn as nn
    class LinearModel(nn.Module):
        def __init__(self, h_dims):
            super(LinearModel,self).__init__()

            models = []
            for i in range(len(h_dims) - 1):
                models.append(nn.Linear(h_dims[i], h_dims[i + 1]))
                if i != len(h_dims) - 2:
                    models.append(nn.ReLU()) 
            self.models = nn.Sequential(*models)
        def forward(self, X):
            return self.models(X)
    
    test_sample_pool = [LinearModel([10,10]) for i in range(10)]
    a = ServerAggregator()
    server = serverSimulator(a)
    ab = server.download_model()
    print(ab)
    for i in test_sample_pool:
        upload_param = {'state_dict': i.state_dict()}
        server.upload_model(upload_param)


    #-----------Unit test for client.py----------------
    #     config = {
    #     'client_id': 1,
    #     'device' : 'cuda',
    #     'lr': 0.001,
    #     'weight_decay' : 0,
    #     'batch_size': 64
    # }

    # train_dataset = FashionMNIST('./data',train=True,download=True,transform=ToTensor())
    # test_dataset = FashionMNIST('./data',train=False,download=True,transform=ToTensor())
    # train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'])
    # class GNN_model(nn.Module):
    #     def __init__(self):
    #         super(GNN_model,self).__init__()
    #         self.flatten = nn.Flatten()
    #         self.linear_relu_stack = nn.Sequential(
    #             nn.Conv2d(in_channels=1,out_channels=4,kernel_size=4,padding=2),
    #             nn.ReLU(),
    #             nn.MaxPool2d(kernel_size=2,stride=2),
    #             nn.Conv2d(in_channels=4,out_channels=8,kernel_size=4,padding=2),
    #             nn.ReLU(),
    #             nn.MaxPool2d(kernel_size=2,stride=2),
    #             nn.Flatten(),
    #             nn.Linear(8*7*7,10),
    #         )
    #     def forward(self,x):
    #         logits = self.linear_relu_stack(x)
    #         return logits

    # model = GNN_model().to(config['device'])
    # print(model)

    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD
    # trainer = fedproxTrainer(model=model,
    #                          dataloader=train_dataloader,
    #                          criterion=loss_fn,
    #                          optimizer=optimizer,
    #                          config=config)