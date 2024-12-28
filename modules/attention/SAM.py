import torch
from torch import nn
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
 
if __name__ == '__main__':
    SA = SpatialAttention(7)
    data_in = torch.randn(8,32,300,300)
    data_out = SA(data_in)
    print(data_in.shape)  # torch.Size([8, 32, 300, 300])
    print(data_out.shape)  # torch.Size([8, 1, 300, 300])
 