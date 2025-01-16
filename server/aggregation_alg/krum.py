from ..base.baseAggregator import ServerAggregator
import numpy as np

class krumAggregator(ServerAggregator):
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
            
        # Krum algorithm implementation
        if isinstance(raw_client_model_or_grad_list[0], dict):
            # Convert dict models to numpy arrays
            models = [np.array(list(model.values())) for model in raw_client_model_or_grad_list]
            
            # Calculate pairwise distances
            n = len(models)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    distances[i,j] = np.linalg.norm(models[i] - models[j])
                    distances[j,i] = distances[i,j]
                    
            # Find the model with minimal sum of distances to its n-f-2 nearest neighbors
            f = n // 4  # assuming at most f Byzantine clients
            scores = []
            for i in range(n):
                dists = np.sort(distances[i])
                scores.append(np.sum(dists[1:n-f]))
                
            selected_idx = np.argmin(scores)
            return raw_client_model_or_grad_list[selected_idx]
            
        elif isinstance(raw_client_model_or_grad_list[0], list):
            # For gradient lists
            grads = [np.array(grad) for grad in raw_client_model_or_grad_list]
            n = len(grads)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    distances[i,j] = np.linalg.norm(grads[i] - grads[j])
                    distances[j,i] = distances[i,j]
                    
            f = n // 4
            scores = []
            for i in range(n):
                dists = np.sort(distances[i])
                scores.append(np.sum(dists[:n-f-2]))
                
            selected_idx = np.argmin(scores)
            return raw_client_model_or_grad_list[selected_idx]
