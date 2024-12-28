import torch
import torch.nn as nn


class PatchSpatialAttention(nn.Module):
    def __init__(self, inp, oup, channel, h, w, k=3):
        super(PatchSpatialAttention, self).__init__()
        self.h = h
        self.w = w
        self.channel = channel
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(channel, channel, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, H, W = x.size()
        h_factor, w_factor = H // self.h, W // self.w

        # 将特征图划分成小块
        y = x.view(batch_size, num_channels, h_factor, self.h, w_factor, self.w)
        y = y.permute(0, 1, 2, 4, 3, 5).contiguous()
        y = y.view(batch_size, num_channels ,h_factor * w_factor, self.h, self.w)

        # 应用自适应平均池化
        y = self.avgpool(y).view(batch_size, num_channels ,h_factor * w_factor).contiguous()

        # 应用1D卷积和sigmoid函数计算注意力权值
        y = self.conv(y)
        y = self.sigmoid(y)

        # 将每个小块的权值与原特征图上的相应元素相乘
        y = y.view(batch_size, num_channels, h_factor * w_factor, 1, 1)
        y = y.expand(batch_size, num_channels, h_factor * w_factor, self.h, self.w)
        y = y.reshape(batch_size, num_channels, h_factor , w_factor, self.h, self.w)
        y = y.permute(0, 1, 2, 4, 3, 5).contiguous()
        y = y.view(batch_size, num_channels,h_factor*self.h,w_factor*self.w)

        x = x * y

        return x