
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import copy
import os 

class AdaptiveBinarize(nn.Module):

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(AdaptiveBinarize, self).__init__()
        self.thresholds = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        
        # max_values = nn.MaxPool2d(kernel_size=(x.shape[2],x.shape[3]))(x)
        # mean_values = nn.AvgPool2d(kernel_size=(x.shape[2],x.shape[3]))(x)
        max_values, _ = torch.max(x, dim=2, keepdim=True)
        max_values, _ = torch.max(max_values, dim=3, keepdim=True)
        # print(max_values.shape)
        min_values, _ = torch.min(x, dim=2, keepdim=True)
        min_values, _ = torch.min(min_values, dim=3, keepdim=True)
        # print(min_values.shape)
        binary_map = torch.where(x > self.thresholds, max_values, min_values)
        return binary_map 


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
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
        g = C // n_div
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
    """Constructs a ECA module.
    source: https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class NormedLinear(nn.Module):                                                                        

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        return cosine


class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes=365, linear_type='Default'):
        super(Classifier, self).__init__()
    
        if linear_type == 'Norm':
            self.fc = NormedLinear(feat_in, num_classes)
        elif linear_type == 'Default':
            self.fc = nn.Linear(feat_in, num_classes)  
        else:
            raise NotImplementedError("Error:Linear {} is not implemented! Please re-choose linear type!".format(linear_type))
    
    def forward(self, x, get_feat=False):
        score = self.fc(x)
        if get_feat == True:
            out = dict()  
            out['feature'] = x
            out['score'] = score
        else:
            out = score

        return out

class Multi_FLENet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 img_size=128,
                 num_classes=8105,
                 embed_dim=20,
                 local_depths=(2,2),
                 global_depths=(6,2),
                 mlp_ratio=2.,
                 local_n_div=4,    # 将特征图按通道方向划分为4部分,只在第一部分进行卷积操作
                 global_n_div=12,  # 将特征图按通道方向划分为12部分,在前四部分分别进行向下，向上、向左、向右的平移操作
                 patch_size=2,     # 初始嵌入层
                 patch_stride=2, 
                 patch_size2=2,    # 阶段内子层
                 patch_stride2=2,
                 group_num = 4,
                 classes_per_group = [2000,2000,2000,2105],
                 patch_norm=True,
                 feature_dim=1280,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='GN1',
                 act_layer='GELU',
                 attention='ECA',
                 Pretraining=False,
                 levels_training = False,
                 multi_head_training = False,
                 init_cfg=None,
                 classifier = True,
                 use_norm= False, 
                 use_noise = False,
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

        self.num_classes = num_classes
        self.group_num = group_num
        self.num_local_stages = len(local_depths)
        self.num_global_stages = len(global_depths)
        self.local_n_div = local_n_div
        self.global_n_div = global_n_div
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.last_channels = int(embed_dim * 2 ** (self.num_global_stages + self.num_global_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.local_depths = local_depths
        self.global_depths = global_depths
        self.Pretraining=Pretraining
        self.levels_training = levels_training
        self.multi_head_training = multi_head_training
        self.classes_per_group = classes_per_group
        self.classifier = classifier
        self.use_norm = use_norm
        self.use_noise = use_noise
        
        
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
        

        self.binary = AdaptiveBinarize(embed_dims=embed_dim)
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
        
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.last_channels, feature_dim, 1, bias=False),
            act_layer()
        )

        # 多个分类头用于识别不同等级的汉字
        # if self.classifier:
        #     if use_norm:
        #         self.linear = NormedLinear(64, num_classes)
        #     else:
        #         self.linear = nn.Linear(64, num_classes)
        
        self.multi_classifier = nn.ModuleList([nn.Linear(feature_dim, classes_per_group[i]+1) for i in range(self.group_num)])
        self.multi_cosine_classifier = nn.ModuleList([NormedLinear(feature_dim, classes_per_group[i]+1) for i in range(self.group_num)])
        

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feature_dim,num_classes)
        
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

    def forward(self, x, group_labels=None,labels=None,train=True):
        x = self.patch_embed(x)
        # x = self.binary(x)
        x = self.pos_drop(x)
        for layer in self.local_layers:
            x = layer(x)
        for layer in self.interactive_layers:
            x = layer(x)
        for layer in self.global_layers:
            x = layer(x)
        if train:
            if group_labels==None:
                print("训练时需要将数据的组标签传入！")
                return False
            if self.Pretraining:  # 在整个数据集上训练一个分类头
                x = self.avgpool_pre_head(x)  # B C 1 1
                x = torch.flatten(x, 1)
                x = self.head(x)
                return x
            else:
                multi_logits = []
                labels_per_group = []
                feats = self.avgpool_pre_head(x)
                feats = torch.flatten(feats, 1)

                for g in range(self.group_num):
                    pos_mask = group_labels >= g
                    if pos_mask.sum()>0:
                        pos_mask_2 = group_labels > g
                        g_logits = self.multi_classifier[g](feats[pos_mask])
                        g_logits = self.pos_drop(g_logits)
                        multi_logits.append(g_logits)
                        mask_labels = labels.clone().detach()
                        mask_labels[pos_mask_2] = self.classes_per_group[g]
                        g_labels = mask_labels[pos_mask]
                        labels_per_group.append(g_labels)
                
                return multi_logits,labels_per_group
        else:
            feats = self.avgpool_pre_head(x)
            feats = torch.flatten(feats, 1)
            

            preds_labels = torch.empty(feats.size(0), dtype=torch.long, device=feats.device)
            preds_groups = torch.zeros(feats.size(0), dtype=torch.long, device=feats.device)
            pos_mask = torch.ones(feats.size(0), dtype=torch.bool, device=feats.device)
            for g in range(self.group_num):
                logits = self.multi_cosine_classifier[g](feats[pos_mask])
                preds = torch.max(logits, dim=1)[1]
                preds_labels[pos_mask] = preds
                pos_mask = preds_labels==self.classes_per_group[g]
                if pos_mask.sum()==0:
                    break
                preds_groups[pos_mask] = g+1

            pred = torch.stack([preds_groups, preds_labels], dim=1)

            return pred

def Multi_FLENet_T0(num_classes):
    model = Multi_FLENet(in_chans=3,
            img_size=128,
            num_classes=num_classes,
            embed_dim=20,
            local_depths=(1,4),
            global_depths=(6,2),
            attention='ECA')
    return model

def Multi_FLENet_T1(num_classes):
    model = Multi_FLENet(in_chans=3,
            img_size=128,
            num_classes=num_classes,
            embed_dim=40,
            local_depths=(1,4),
            global_depths=(6,2),
            attention='ECA')
    return model

def Multi_FLENet_T2(num_classes):
    model = Multi_FLENet(in_chans=3,
            img_size=128,
            num_classes=num_classes,
            embed_dim=60,
            local_depths=(1,4),
            global_depths=(6,2),
            attention='ECA')
    return model



if __name__ == '__main__':
    x = torch.randn(5, 3, 128, 128)
    groups = torch.tensor([1,1,1,2,2])
    labels = torch.tensor([11,12,51,33,12])

    model = Multi_FLENet(num_classes=8105)
    # multi_logits,labels_per_group = model(x,groups,labels,train=True)
    pred = model(x,train=False)
    # print(labels_per_group)
    print(pred)
    5