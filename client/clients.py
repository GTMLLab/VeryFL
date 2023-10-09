#This implement the client class aims to intergate the trainging process.
from abc import abstractmethod
from client.base.baseTrainer import BaseTrainer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch



class client:
    '''
    Client base class for a client in federated learning

    1.Just receive a cid for the uique toked when initialize
    2.download() method to get global model form the server
    3.Use train() to produce training 
    4.Test is no need in this senario 
    '''
    def __init__(
        self,
        client_id:str,
        dataloader:DataLoader,
        model:nn.Module,
        trainer: BaseTrainer,
        num_steps = 1,
        args = {},
        #args is a dict
    )->None:
        self.model = model
        self.client_id = client_id
        self.dataloader = dataloader
        self.trainer = trainer
        self.args = args
    def set_model(self,model:nn.Module):
        self.model = model
    
    @abstractmethod
    def train(self):
        return

class BaseClient(client):
    def __init__():
        super().__init__()
    def train(self):
        cal = self.trainer(self.model,self.dataloader,torch.nn.CrossEntropyLoss(),torch.optim.SGD,self.args)
        cal.train(self.num_steps)
        return