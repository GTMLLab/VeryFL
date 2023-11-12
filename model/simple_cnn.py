#use with FashionMnist
import torch.nn as nn
class simpleCNN(nn.Module):
    def __init__(self,class_num):
        super(simpleCNN,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=4,kernel_size=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=4,out_channels=8,kernel_size=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(8*7*7,class_num),
        )
    def forward(self,x):
        logits = self.linear_relu_stack(x)
        return logits

def get_simple_cnn(class_num): return simpleCNN(class_num)