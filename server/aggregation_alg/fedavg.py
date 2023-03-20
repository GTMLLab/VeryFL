'''
请实现一个fedavg的aggregator，通过继承bashAggregator
即你需要实现虚函数_aggregation_alg() 
如有不清楚的请查看ServerAggregator中的注释，或联系
'''
from ..base.baseAggregator import ServerAggregator

class fedavgAggregator(ServerAggregator):
    def __init__():
        pass
    
    def _aggregate_alg(self, raw_client_model_or_grad_list):
        return super()._aggregate_alg(raw_client_model_or_grad_list)
    
    
if __name__=='__main__':
    #write your unit test here.
    pass
    