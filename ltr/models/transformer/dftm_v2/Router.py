import torch
import torch.nn as nn
import torch.nn.functional as F

def activateFunc(x):
    x = torch.tanh(x)
    return F.relu(x)

class Router(nn.Module):
    def __init__(self, num_out_path, embed_size):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        #self.conv_pool = nn.Linear(embed_size*36*36, embed_size)
        self.mlp = nn.Sequential(nn.Linear(embed_size*2, embed_size), 
                                    nn.ReLU(True), 
                                    nn.Linear(embed_size, num_out_path))
        self.init_weights()

    def init_weights(self):
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        #x = x.reshape(x.shape[0],-1)
        avg_out = self.avgpool(x)
        max_out = self.maxpool(x)
        f_out = torch.cat([max_out,avg_out],1)
        x = f_out.contiguous().view(f_out.size(0), -1)
        x = self.mlp(x)
        soft_g = activateFunc(x)
        return soft_g
