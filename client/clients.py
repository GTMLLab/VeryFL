#This implement the client class aims to intergate the trainging process.
from abc import abstractmethod
from client.base.baseTrainer import BaseTrainer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from copy import deepcopy
from typing import OrderedDict

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
        client_id: str,
        dataloader: DataLoader,
        model: nn.Module,
        trainer: BaseTrainer,
        num_steps: int = 1,
        args: dict = {},
    ) -> None:
        self.model = deepcopy(model)
        self.client_id = client_id
        self.dataloader = dataloader
        self.trainer = trainer
        self.args = args
        self.num_steps = num_steps
    
    def set_model(self, model: nn.Module) -> None:
        self.model = deepcopy(model)
        
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict=state_dict)
        
    def get_model_state_dict(self) -> OrderedDict:
        return self.model.state_dict()
    
    @abstractmethod
    def train(self):
        return

class BaseClient(client):
    def train(self):
        cal = self.trainer(self.model,self.dataloader,torch.nn.CrossEntropyLoss(),torch.optim.SGD,self.args)
        cal.train(self.num_steps)
        return