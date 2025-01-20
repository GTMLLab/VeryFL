import logging
from server.base.baseAggregator import ServerAggregator
from server.aggregation_alg.fedavg import fedavgAggregator
from server.aggregation_alg.fedproto import fedprotoAggregator
from client.base.baseTrainer import BaseTrainer
from client.trainer.normalTrainer import normalTrainer
from client.clients import Client, BaseClient
logger = logging.getLogger(__name__)

'''
FL Algorithm Info:
1. Write server derived from ServerAggregator
2. Write trainer derived from BaseTrainer
3. Write client(may not used) 
'''

class Algorithm:
    def __init__(self, 
                 server:  ServerAggregator  = None, 
                 trainer: BaseTrainer       = None,
                 client:  Client            = None):
        self.server  = server if server is not None else fedavgAggregator
        self.trainer = trainer if trainer is not None else normalTrainer
        self.client  = client if client is not None else BaseClient
    
    def get_server(self):
        return self.server
    
    def get_trainer(self):
        return self.trainer
    
    def get_client(self):
        return self.client
    

class FedAvg(Algorithm):
    def __init__(self, client=None):
        super(FedAvg, self).__init__(client=client)

from client.trainer.fedproxTrainer import fedproxTrainer
class FedProx(Algorithm):
    def __init__(self):
        super(FedProx, self).__init__(trainer=fedproxTrainer)

from client.trainer.fedprotoTrainer import fedprotoTrainer
class FedProto(Algorithm):
    def __init__(self):
        super(FedProto, self).__init__(trainer=fedprotoTrainer, server=fedprotoAggregator)

from client.trainer.SignTrainer import SignTrainer
from client.clients import SignClient  
class FedIPR(Algorithm):
    def __init__(self):
        super(FedIPR, self).__init__(trainer=SignTrainer, client=SignClient)
        
        

__all__ = [
    "FedAvg",
    "FedProx",
    "FedProto",
    "FedIPR",   
]