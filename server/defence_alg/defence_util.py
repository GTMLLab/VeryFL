# 这个文件下将要实现一些常用的服务端安全聚合策略
# 最终util会用在baseAggregator的on_before_aggregation或者on_after_aggregation这个方法中
# 即你要在下面的serverDefender类中实现一些简单的服务端防御算法（一般github上有实现的话就可以不用从0开始写）

from typing import List, Tuple, Dict, OrderedDict

class serverDefender:
    def __init__():
        pass
    
    # 接受一个由各个客户端模型组成的模型list，
    # 通过特定算法将这些可能含有恶意梯度的list进行筛选并返回一个清理后的list。
    # 实现5组即可 常见的如Krum等可以找一些baseline进行实现。
    def alg1(self,raw_client_model_or_grad_list: List[OrderedDict]):
        pass
    def alg2(self,raw_client_model_or_grad_list: List[OrderedDict]):
        pass
    #..... alg3,4,5 把名字也改过来
