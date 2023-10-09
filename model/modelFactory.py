from .resnet import ResNet18
from .alexnet import createAlexNet


class modelFactory:
    '''
    Server as an factory for user to get predefined neural network.
    the one is ImageClassification Factory used to construct AlexNet, resnet with the known class_num.
    '''
    def __init__(self,)->None:
        return
    def get_model(self,model,class_num):
        #now model could be: resnet, alexnet, ...
        if model=='resnet':
            return ResNet18(class_num)
        elif model=='alexnet':
            return createAlexNet(class_num)
        else:raise Exception(f"Unrecognized Model")
        

