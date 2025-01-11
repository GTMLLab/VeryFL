from ..base.baseAggregator import ServerAggregator
import copy

class fedprotoAggregator(ServerAggregator):
    def __init__(self):
        super().__init__()
    def _on_before_aggregation():
        pass
    def _on_after_aggregation():
        pass
    def test():
        pass
    def _aggregate_alg(self, raw_client_list=None):
        if(raw_client_list is None): raw_client_list = self.client_pool
        raw_client_model_or_grad_list = [client.model.state_dict() for client in raw_client_list]
        # 实现FedAvg聚合算法
        aggregated_model = None
        num_clients = len(raw_client_list)
        if isinstance(raw_client_model_or_grad_list[0], dict):
            # 客户端模型参数列表
            model_size = len(raw_client_model_or_grad_list[0])  # 假设模型大小相同
            aggregated_model = {}
            for key in raw_client_model_or_grad_list[0].keys():
                sum_values = sum(client_model[key] for client_model in raw_client_model_or_grad_list)
                aggregated_model[key] = sum_values / num_clients
        elif isinstance(raw_client_model_or_grad_list[0], list):
            # 客户端梯度列表
            grads_sum = [sum(grad[i] for grad in raw_client_model_or_grad_list) for i in range(len(raw_client_model_or_grad_list[0]))]
            aggregated_model = [grad_sum / num_clients for grad_sum in grads_sum]
        
        # aggregate prototypes
        updated_protos = {}
        for _, client in enumerate(raw_client_list):
            for key in client.trainer.protos.keys():
                if key in updated_protos.keys():
                    updated_protos[key] += client.trainer.protos[key] / num_clients
                else:
                    updated_protos[key] = client.trainer.protos[key] / num_clients

        for _, client in enumerate(raw_client_list):
            for key in updated_protos.keys():
                client.trainer.protos[key] = updated_protos[key].clone()

        return aggregated_model
