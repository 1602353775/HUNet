from torch import nn
import torch
 
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        # return x * y
 
 
if __name__ == '__main__':
    torch.manual_seed(seed=20200910)
    data_in = torch.randn(8,32,300,300)
    SE = SELayer(32) 
    data_out = SE(data_in)
    print(data_in.shape)  # torch.Size([8, 32, 300, 300])
    print(data_out.shape)  # torch.Size([8, 32, 300, 300])
    
    
    
    
    
    
    