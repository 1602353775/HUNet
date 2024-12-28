import torch
from torch import nn
 
class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
 
		self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
		self.sigmoid = nn.Sigmoid()
 
	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		return self.sigmoid(out)
 
 
if __name__ == '__main__':
    CA = ChannelAttention(32)
    data_in = torch.randn(8,32,300,300)
    data_out = CA(data_in)
    print(data_in.shape)  # torch.Size([8, 32, 300, 300])
    print(data_out.shape)  # torch.Size([8, 32, 1, 1])
 