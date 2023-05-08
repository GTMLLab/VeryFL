'''
请实现一个fedavg的aggregator，通过继承bashAggregator
即你需要实现虚函数_aggregation_alg() 
如有不清楚的请查看ServerAggregator中的注释，或联系
'''
from ..base.baseAggregator import ServerAggregator

class fedavgAggregator(ServerAggregator):
    def __init__(self):
        super().__init__()

    def _aggregate_alg(self, raw_client_model_or_grad_list):
        # 实现FedAvg聚合算法
        aggregated_model = None
        if isinstance(raw_client_model_or_grad_list[0], dict):
            # 客户端模型参数列表
            num_clients = len(raw_client_model_or_grad_list)
            model_size = len(raw_client_model_or_grad_list[0])  # 假设模型大小相同
            aggregated_model = {}
            for key in raw_client_model_or_grad_list[0].keys():
                sum_values = sum(client_model[key] for client_model in raw_client_model_or_grad_list)
                aggregated_model[key] = sum_values / num_clients
        elif isinstance(raw_client_model_or_grad_list[0], list):
            # 客户端梯度列表
            num_clients = len(raw_client_model_or_grad_list)
            grads_sum = [sum(grad[i] for grad in raw_client_model_or_grad_list) for i in range(len(raw_client_model_or_grad_list[0]))]
            aggregated_model = [grad_sum / num_clients for grad_sum in grads_sum]

        return aggregated_model


if __name__ == '__main__':
    # 单元测试
    client1_model = {'weight': 0.5, 'bias': 0.2}
    client2_model = {'weight': 0.3, 'bias': 0.1}
    client3_model = {'weight': 0.6, 'bias': 0.3}

    aggregator = fedavgAggregator()
    aggregated_model = aggregator._aggregate_alg([client1_model, client2_model, client3_model])

    print(aggregated_model)
