from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, OrderedDict
class ServerAggregator(ABC):
    """Abstract base class for federated learning trainer.
    
    :method: on_before_aggregation: do some operation to simulate the communiation phase 
                                    between client and the server.
             on_after_aggregation:  do some operation to simulate the manipulation 
                                    when server perpare the send the aggregated model to the client.
             _aggregation_alg: the main aggregation federated learning algoritm.
    """
     
    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, aggregator_id):
        self.id = aggregator_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass
    
    @abstractmethod
    def _aggregate_alg(self,raw_client_model_or_grad_list):
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
    def on_before_aggregation(
        self, raw_client_model_or_grad_list: List[OrderedDict]
    ) -> List[OrderedDict]:
        '''
        In this class, we may simulate the attackers to posion the model gradient during the upload stage.
        And we do some thing to poison the gredient, if no poision method is used, this function leaves as hollow.
        '''
        pass

    def aggregate(self, raw_client_model_or_grad_list: List[OrderedDict]) -> OrderedDict:
        return self._aggregate_alg(raw_client_model_or_grad_list)
        return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)

    def on_after_aggregation(self, aggregated_model_or_grad: OrderedDict) -> OrderedDict:
        '''
        to do some coupled operation with the on before the aggregation.
        like the dp-encode or model compression stratgy.
        If we donnot use this advanced method, this function will be leaved empty as well.
        '''
        return aggregated_model_or_grad

    @abstractmethod
    def test(self, test_data, device, args):
        pass
    
    @abstractmethod
    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        pass