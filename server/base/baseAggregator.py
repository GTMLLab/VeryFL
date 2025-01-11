from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, OrderedDict
from client.clients import Client
class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer.
    
    :method: on_before_aggregation: do some operation to simulate the communiation phase 
                                    between client and the server.
             on_after_aggregation:  do some operation to simulate the manipulation 
                                    when server perpare the send the aggregated model to the client.
             _aggregation_alg: the main aggregation federated learning algoritm.
    """
     
    def __init__(self, model=None, args=None):
        self.model = model
        self.id = 0
        self.args = args
        self.client_pool = None
    def set_id(self, aggregator_id):
        self.id = aggregator_id
    def receive_upload(self, client_pool: List[Client]):
        self.client_pool = client_pool
        
    @abstractmethod
    def _aggregate_alg(self, raw_client_list: List[Client]=None):
        '''
        This method complement the aggregation method, 
        like fedavg and some aggregate operation done in server side,
        but some alg like fedprox which need the cooperation with the client side 
        will be complemented locally. 
        
        :param format: the function will receive a list of clients'state_dict (the model param in torch may be ordered dict pay attention)
        :return format: the fucntion will return an aggregated state_dict which suit the local model.
        '''
        pass
    
        
    @abstractmethod
    def _on_before_aggregation(
        self, raw_client_model_or_grad_list: List[OrderedDict]
    ) -> List[OrderedDict]:
        '''
        In this class, we may simulate the attackers to posion the model gradient during the upload stage.
        And we do some thing to poison the gredient, if no poision method is used, this function leaves as hollow.
        '''
        pass

    def aggregate(self, raw_client_list: List[Client]=None) -> OrderedDict:
        if(raw_client_list is None): raw_client_list = self.client_pool
        return self._aggregate_alg(raw_client_list)
        #return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)

    def _on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        '''
        to do some coupled operation with the on before the aggregation.
        like the dp-encode or model compression stratgy.
        If we donnot use this advanced method, this function will be leaved empty as well.
        '''
        return aggregated_model_or_grad

    @abstractmethod
    def test(self, test_data, device, args):
        pass
    