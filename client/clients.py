#This implement the client class aims to intergate the trainging process.
from abc import abstractmethod
import logging

from client.base.baseTrainer import BaseTrainer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from copy import deepcopy
from typing import OrderedDict

logger = logging.getLogger(__name__)

class Client:
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
        args: dict = {},
        test_dataloader: DataLoader = None,
    ) -> None:
        self.model = deepcopy(model)
        self.client_id = client_id
        self.dataloader = dataloader
        self.trainer = trainer
        self.args = args
        self.test_dataloader = test_dataloader
    
    def set_model(self, model: nn.Module) -> None:
        self.model = deepcopy(model)
        
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict=state_dict)
        
    def get_model_state_dict(self) -> OrderedDict:
        return self.model.state_dict()
    
    def test(self, epoch: int) -> dict:
        '''
        return dict
        : client_id
          epoch
          loss
          acc
        '''
        #test routine for image classification 
        if (self.test_dataloader == None):
            logger.warn("No test data")
            return 
        self.model.eval()
        total_loss = 0
        correct = 0
        num_data = 0
        predict_label = torch.tensor([]).to(self.args['device'])
        true_label = torch.tensor([]).to(self.args['device'])
        for batch_id, batch in enumerate(self.test_dataloader):
            data, targets = batch
            data, targets = data.to(self.args['device']), targets.to(self.args['device'])
            true_label = torch.cat((true_label, targets), 0)
            output = self.model(data)
            total_loss += torch.nn.functional.cross_entropy(output, targets,
                                                            reduction='sum').item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            predict_label = torch.cat((predict_label, pred), 0)
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            num_data += output.size(0)
        acc = 100.0 * (float(correct) / float(num_data))
        total_l = total_loss / float(num_data)
        ret = dict()
        ret['client_id'] = self.client_id
        ret['epoch'] = epoch
        ret['loss'] = total_l
        ret['acc'] = acc
        logger.info(f"client id {self.client_id} with inner epoch {ret['epoch']}, Loss: {total_l}, Acc: {acc}")
        return ret
    @abstractmethod
    def train(self, epoch: int):
        return

class BaseClient(Client):
    def train(self, epoch: int):    
        cal = self.trainer(self.model,self.dataloader,torch.nn.CrossEntropyLoss(),self.args)
        avg_loss = cal.train(self.args.get('num_steps'))
        logger.info(f"Epoch: {epoch}, client id {self.client_id}, Loss: {avg_loss} ")
        return
    

class SignClient(Client):
    def sign_test(self, kwargs, ind):
        self.model.eval()
        avg = 0
        count = 0 
        with torch.no_grad():
            return