'''
在这里调研实现一些常用的模型压缩算法，如剪枝，量化，等
'''
from collections import OrderedDict

'''
量化
'''
def quantify_encode(model_state_dict:OrderedDict) : #返回类型应该也是一个orderedDict
    '''
    在这里实现编码，或者叫压缩
    :param: model_state_dict
            看看是否还有其他需要传入的参数，应该还需要设置压缩的程度
    '''
    
def quantify_encode(model_compressed_state_dict:OrderedDict):
    '''
    在这里实现解码，或者解压缩
    :param: model_compressed_state_dict
            看看是否还有其他需要传入的参数,应该还需要知道原来的模型结构和参数
    '''
    
#剪枝有点类似pytorch中的dropout可以去看看
'''
剪枝，类似上面的接口形式和函数规则
'''

if __name__ == '__main__':
    #write your unit test here.
    pass