from client.trainer.fedproxTrainer import fedproxTrainer
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch
config = {
    'client_id': 1,
    'device' : 'cuda',
    'lr': 0.001,
    'weight_decay' : 0,
    'batch_size': 64
}

train_dataset = FashionMNIST('./data',train=True,download=True,transform=ToTensor())
test_dataset = FashionMNIST('./data',train=False,download=True,transform=ToTensor())
train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'])
class GNN_model(nn.Module):
    def __init__(self):
        super(GNN_model,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=4,kernel_size=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=4,out_channels=8,kernel_size=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(8*7*7,10),
        )
    def forward(self,x):
        logits = self.linear_relu_stack(x)
        return logits

model = GNN_model().to(config['device'])
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD
trainer = fedproxTrainer(model=model,
                         dataloader=train_dataloader,
                         criterion=loss_fn,
                         optimizer=optimizer,
                         config=config)


if __name__ =="__main__":
    
    trainer.train(10)
