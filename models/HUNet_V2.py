
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os 
import math
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Parameter
from models.Head import DynamicMargin_Unified_Cross_Entropy_Loss_V2, Unified_Cross_Entropy_Loss

class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3,stride=2, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchEmbed_g(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int, tuple): Image size.
        patch_size (int, tuple): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=1):
        """ We use GroupNorm (group = 1) to approximate LayerNorm
        for [N, C, H, W] layout"""
        super(GroupNorm, self).__init__(num_groups, num_channels)


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class LocalMlpBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class LocalStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 input_resolution,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super(LocalStage, self).__init__()
        
        self.dim = dim
        self.depth = depth
        self.input_resolution = input_resolution
        
        blocks_list = [
            LocalMlpBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"depth={self.depth}"


class PatchMerging(nn.Module):
    # 先合并降维，再归一化；
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class PatchMerging_s(nn.Module):
    # 先归一化，再合并降维；
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, (2, 2), stride=2, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class GlobalMlpBlock(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        """ MLP network in FFN. By default, the MLP is implemented by
        nn.Linear. However, in our implementation, the data layout is
        in format of [N, C, H, W], therefore we use 1x1 convolution to
        implement fully-connected MLP layers.
        Args:
            in_features (int): input channels
            hidden_features (int): hidden channels, if None, set to in_features
            out_features (int): out channels, if None, set to in_features
            act_layer (callable): activation function class type
            drop (float): drop out probability
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ShiftViTBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=12,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 input_resolution=None):
        """ The building block of Shift-ViT network.
        Args:
            dim (int): feature dimension
            n_div (int): how many divisions are used. Totally, 4/n_div of
                channels will be shifted.
            mlp_ratio (float): expand ratio of MLP network.
            drop (float): drop out prob.
            drop_path (float): drop path prob.
            act_layer (callable): activation function class type.
            norm_layer (callable): normalization layer class type.
            input_resolution (tuple): input resolution. This optional variable
                is used to calculate the flops.
        """
        super(ShiftViTBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GlobalMlpBlock(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.n_div = n_div

    def forward(self, x):
        x = self.shift_feat(x, self.n_div)
        shortcut = x
        x = shortcut + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"shift percentage={4.0 / self.n_div * 100}%."

    @staticmethod
    def shift_feat(x, n_div):
        B, C, H, W = x.shape
        # g = C // n_div
        g = torch.div(C, n_div, rounding_mode='trunc')
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down

        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return out


class GlobalShiftStage(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 n_div=12,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=None,
                 norm_layer=None,
                 act_layer=nn.GELU):

        super(GlobalShiftStage, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # build blocks
        self.blocks = nn.ModuleList([
            ShiftViTBlock(dim=dim,
                          n_div=n_div,
                          mlp_ratio=mlp_ratio,
                          drop=drop,
                          drop_path=drop_path[i],
                          norm_layer=norm_layer,
                          act_layer=act_layer,
                          input_resolution=input_resolution)
            for i in range(depth)
        ])


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}," \
               f"input_resolution={self.input_resolution}," \
               f"depth={self.depth}"


class eca_layer(nn.Module):
    """ECA (Efficient Channel Attention) 模块实现
    来源: https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    
    参数说明:
        channel: 输入特征图的通道数（C）
        k_size: 1D卷积的核大小（默认3，必须是奇数）
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        
        # 全局平均池化（将H×W的空间信息压缩为1×1）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        
        # 1D卷积（用于捕获跨通道交互）
        # 输入/输出通道数均为1，卷积核大小可调，padding保证长度不变
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                             padding=(k_size - 1) // 2, bias=False)  
        
        # Sigmoid激活函数（生成0~1的注意力权重）
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        # 输入x形状: [batch_size, C, H, W]
        
        # 步骤1: 全局空间信息描述（压缩空间维度）
        # y形状: [batch_size, C, 1, 1]
        y = self.avg_pool(x)  
        
        # 步骤2: 跨通道交互（ECA核心操作）
        # 操作流程: 
        # 1) 移除最后1维 -> [batch_size, C, 1]
        # 2) 转置最后两维 -> [batch_size, 1, C]（符合Conv1d输入格式）
        # 3) 1D卷积处理 -> [batch_size, 1, C]
        # 4) 转置恢复 -> [batch_size, C, 1]
        # 5) 添加最后1维 -> [batch_size, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # 1D卷积处理
        y = y.transpose(-1, -2).unsqueeze(-1)  # 恢复形状
        
        # 步骤3: 生成注意力权重（0~1范围）
        y = self.sigmoid(y)  # 形状保持[batch_size, C, 1, 1]
        
        # 步骤4: 对原始特征图进行通道加权
        # expand_as(x)将权重广播为[batch_size, C, H, W]
        return x * y.expand_as(x)  


class HUNet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 img_size=128,
                 embed_dim=96,
                 local_depths=(2,2),
                 global_depths=(6,2),
                 mlp_ratio=4.,
                 local_n_div=4,    # 将特征图按通道方向划分为4部分,只在第一部分进行卷积操作
                 global_n_div=12,  # 将特征图按通道方向划分为12部分,在前四部分分别进行向下，向上、向左、向右的平移操作
                 patch_size=2,     # 初始嵌入层
                 patch_stride=2, 
                 patch_size2=2,    # 阶段内子层
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='GN1',
                 act_layer='GELU',
                 attention='ECA',
                 init_cfg=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()
        
        # 
        assert norm_layer in ('GN1', 'BN')
        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'GN1':
            norm_layer = partial(GroupNorm, num_groups=1)
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError
        

        assert attention in ('SE', 'SK', 'CBAM', 'CAM', 'SAM','SBAM', 'CA', 'ECA', 'PA', 'RFAConv','Moga')

        self.num_local_stages = len(local_depths)
        self.num_global_stages = len(global_depths)
        self.local_n_div = local_n_div
        self.global_n_div = global_n_div
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_global_stages + self.num_global_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.local_depths = local_depths
        self.global_depths = global_depths
        
        
        # (b,3,128,128)
        # 将图像分割成不重叠的小块
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        # (b,96,64,64)
        
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则  从0开始到drop_path_rate结束，生成了sum(depths)个数值。最后，将生成的所有数值转化成标量值，并存储在列表 dpr 中，以备后续计算 dropout 时使用。
        l_dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(local_depths))]  # 长度为 4 

        # 局部特征提取阶段，去除背景噪声，减弱文字颜色和纹理、凸显文字区域
        self.local_input_size = img_size//patch_size
        self.local_layers = nn.ModuleList()
        for i_stage in range(self.num_local_stages):
            Local_stage = LocalStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=self.local_n_div,
                               depth=local_depths[i_stage],
                               input_resolution = (self.local_input_size//2**i_stage,
                                                   self.local_input_size//2**i_stage),
                               mlp_ratio=self.mlp_ratio,
                               drop_path=l_dpr[sum(local_depths[:i_stage]):sum(local_depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            self.local_layers.append(Local_stage)
            

            # 小块合并层-降维
            if i_stage < self.num_local_stages - 1:
                self.local_layers.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )
        # (b,96,64,64)*2   (b,192,32,32)*2
        
        # 局部特征提取和全局结构特征提取的交互阶段，将最重要的局部特征送入下一阶段
        self.interactive_layers = nn.ModuleList()
        attention_layer = eca_layer(channel=embed_dim * 2 ** (self.num_local_stages-1), k_size=3)
        
        self.interactive_layers.append(attention_layer)
        Downsampling = PatchMerging(patch_size2=patch_size2,
                                    patch_stride2=patch_stride2,
                                    dim=int(embed_dim * 2 ** (self.num_local_stages-1)),
                                    norm_layer=norm_layer)
        self.interactive_layers.append(Downsampling)
        # (b,384,16,16)

        
        # 全局结构特征提取阶段，通过平移操作使得模型学习到一个汉字的整体结构和各个构件的组合关系
        g_dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(global_depths))]
        
        self.global_layers = nn.ModuleList()
        self.global_input_size = self.local_input_size//2**self.num_local_stages
        for i_stage in range(self.num_global_stages):
            global_stage = GlobalShiftStage(dim=int(embed_dim * 2 ** (len(local_depths)+i_stage)),
                               n_div=self.global_n_div,
                               input_resolution=(self.global_input_size//2**i_stage,
                                                 self.global_input_size//2**i_stage),
                               depth=global_depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=g_dpr[sum(global_depths[:i_stage]):sum(global_depths[:i_stage + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer)
            self.global_layers.append(global_stage)
            # 小块合并层-降维
            if i_stage < self.num_global_stages - 1:
                self.global_layers.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** (len(local_depths)+i_stage)),
                                 norm_layer=norm_layer)
                )
        #  (b,384,16,16)*6 逐渐偏移6个像素 (b,768,8,8)*2 逐渐偏移2个像素

        # Classifier head
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
            act_layer()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # (b,768,1,1) --->(b,1280,1,1) --->(b,1280) --->(b,8105)
        
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.local_layers:
            x = layer(x)
        for layer in self.interactive_layers:
            x = layer(x)
        for layer in self.global_layers:
            x = layer(x)
        x = self.avgpool_pre_head(x)  
        feats = torch.flatten(x, 1)
       
        return feats
    

def HUNet_24():
    model = HUNet(in_chans=3,
                 img_size=128,
                 embed_dim=24,
                 local_depths=(2,2),
                 global_depths=(6,2),
                 attention='ECA')
    return model


def HUNet_32():
    model = HUNet(in_chans=3,
                 img_size=128,
                 embed_dim=32,
                 local_depths=(2,2),
                 global_depths=(6,2),
                 attention='ECA')
    return model


def HUNet_36():
    model = HUNet(in_chans=3,
                 img_size=128,
                 embed_dim=36,
                 local_depths=(2,2),
                 global_depths=(6,2),
                 attention='ECA')
    return model


def HUNet_48():
    model = HUNet(in_chans=3,
                 img_size=128,
                 embed_dim=48,
                 local_depths=(2,2),
                 global_depths=(6,2),
                 attention='ECA')
    return model


class HUNet_V2(nn.Module):
    def __init__(self, num_classes=None, mode="train"):
        super(HUNet_V2, self).__init__()
        embedding_size  = 1280
        s               = 64
        self.feature    = HUNet_24()
        self.head = Unified_Cross_Entropy_Loss(in_features=embedding_size, out_features=num_classes, s=s)
                
    def forward(self, x, y = None, mode = "train"):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        if mode == "feature":
            return x
        else:
            outputs = self.head(x, y)
            return outputs


if __name__ == "__main__":

    from thop import profile
    from ptflops import get_model_complexity_info


    def model_thop(model, input_size):
        # 使用thop计算FLOPs和参数量
        input = torch.randn(1, *input_size)  # 创建随机输入
        flops, params = profile(model, inputs=(input,), verbose=False)
        
        print('\n' + '='*50)
        print(f"{'Computational complexity:':<30} {flops/1e9:.2f}G FLOPs")
        print(f"{'Number of parameters:':<30} {params/1e6:.2f}M")
        print('='*50 + '\n')


    def model_ptflops(model,input):
        macs, params = get_model_complexity_info(model, input, as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    def count(model):
        model_thop(model,(3,128,128))
        # model_ptflops(model,(3,128,128))

    model_1 = HUNet_24()
    model_2 = HUNet_32()
    model_3 = HUNet_36()
    model_4 = HUNet_48()
  
    count(model_1)
    count(model_2)
    count(model_3)
    count(model_4)