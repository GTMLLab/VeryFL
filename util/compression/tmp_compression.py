
'''
在这里调研实现一些常用的模型压缩算法，如剪枝，量化，等
'''
from collections import OrderedDict
import torch

'''
量化
'''


def quantify_encode(model_state_dict: OrderedDict):  # 返回类型应该也是一个orderedDict
    '''
    在这里实现编码，或者叫压缩
    :param: model_state_dict
            看看是否还有其他需要传入的参数，应该还需要设置压缩的程度
    '''
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float32:
                value *= 10
                value.clamp_(-128, 127)
                value = value.to(torch.int8)
                model_state_dict[key] = value
    return model_state_dict


def quantify_decode(model_compressed_state_dict: OrderedDict):
    '''
    在这里实现解码，或者解压缩
    :param: model_compressed_state_dict
            看看是否还有其他需要传入的参数,应该还需要知道原来的模型结构和参数
    '''
    for key in model_compressed_state_dict:
        if isinstance(model_compressed_state_dict[key], torch.Tensor) and model_compressed_state_dict[
            key].dtype == torch.int8:
            # 将int8类型的参数张量先除以缩放因子得到float32类型的参数张量
            model_compressed_state_dict[key] = model_compressed_state_dict[key].to(torch.float32) / 10
            # model_compressed_state_dict[key] /= 10.0
    return model_compressed_state_dict


# 剪枝有点类似pytorch中的dropout可以去看看
'''
剪枝，类似上面的接口形式和函数规则
'''

if __name__ == '__main__':
    # write your unit test here.
    t = torch.randn(2, 3)
    test = torch.tensor(t)
    print(test)
    dic = OrderedDict()
    dic['test'] = test
    print(dic.items())
    quantify_encode(dic)

    print("\nafter quantifying:")
    print(dic.items())

    print("\nafter decode: ")
    quantify_decode(dic)
    print(dic.items())
    # y = torch.tensor(dic['test'])
    # print(test, y)
    print("\norigin data:", t, "\ndata after operating", dic['test'])
    print("variance:", torch.nn.functional.mse_loss(t, dic['test']))
    pass