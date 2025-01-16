from ..base.baseAggregator import ServerAggregator
import numpy as np

class medianAggregator(ServerAggregator):
    def __init__(self):
        super().__init__()
        
    def _on_before_aggregation(self):
        pass
        
    def _on_after_aggregation(self):
        pass
        
    def test(self):
        pass
        
    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool
            
        if isinstance(raw_client_model_or_grad_list[0], dict):
            # For dictionary models
            keys = raw_client_model_or_grad_list[0].keys()
            aggregated_model = {}
            
            for key in keys:
                # Get all values for this parameter across clients
                values = [model[key] for model in raw_client_model_or_grad_list]
                # Compute median
                aggregated_model[key] = np.median(values)
                
            return aggregated_model
            
        elif isinstance(raw_client_model_or_grad_list[0], list):
            # For gradient lists
            grads = np.array(raw_client_model_or_grad_list)
            # Compute median along the client axis
            median_grad = np.median(grads, axis=0)
            return median_grad.tolist()
