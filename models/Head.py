import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicMargin_Unified_Cross_Entropy_Loss_V2(nn.Module):
    """
    动态边际统一交叉熵损失函数
    特征：
    - 基于类别样本数的动态边际计算
    - 正负样本差异化的logit调整
    - 内置类别重平衡机制
    - 数值稳定化处理

    数学公式：
    正样本损失：log(1 + exp(-s·(cosθ - m_i) + b))
    负样本损失：log(1 + exp(s·cosθ - b)) * l
    其中m_i = a·n_i^-λ + b，n_i为类别i的样本数

    Args:
        class_count_dict (dict): 类别样本字典，{类别字符串: 样本数}
        in_features (int): 输入特征维度
        out_features (int): 分类类别数
        s (float): 缩放因子，建议>10
        lambda_ (float): 样本数影响系数，控制长尾分布调整强度
        a (float): 动态边际乘数因子
        b (float): 动态边际基数
        l (float): 负样本损失权重
        r (float): 类别重平衡因子，影响偏置初始化
    """
    
    def __init__(self, class_count_dict, in_features=128, out_features=10575, s=64, 
                 lambda_=0.25, a=0.5, b=0.05, l=1.0, r=1.0):
        super().__init__()
        # 参数校验
        if len(class_count_dict) != out_features:
            raise ValueError("class_count_dict必须包含所有输出类别的样本数")
        
        # 核心参数配置
        self.class_count_dict = {str(k): v for k, v in class_count_dict.items()}  # 统一键类型
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.lambda_ = lambda_
        self.a = a
        self.b = b
        self.l = l
        self.r = r

        # 预计算动态边际
        self.register_buffer('m_l', torch.zeros(out_features))
        self._precompute_margins()

        # 可学习参数
        self.bias = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """参数初始化策略"""
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.bias, math.log(self.out_features * self.r * 10))

    def _precompute_margins(self):
        """预计算所有类别的动态边际"""
        with torch.no_grad():
            for idx in range(self.out_features):
                class_name = str(idx)
                count = self.class_count_dict[class_name]
                self.m_l[idx] = self.a * (count ** -self.lambda_) + self.b

    def forward(self, input, label=None):
        """
        前向计算流程

        Args:
            input (Tensor): 输入特征，(batch_size, in_features)
            label (Tensor): 类别标签，(batch_size,)

        Returns:
            tuple: (cosine相似度矩阵, 损失值) 或 (cosine相似度矩阵,)
        """
        # 特征归一化
        input_norm = F.normalize(input, p=2, dim=1, eps=1e-5)
        weight_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-5)
        
        # 计算余弦相似度
        cos_theta = F.linear(input_norm, weight_norm)  # (batch_size, out_features)
        
        if label is None:
            return (cos_theta,)
        
        # 获取设备信息
        device = input.device
        
        # 生成one-hot编码
        one_hot = torch.zeros((label.size(0), self.out_features), 
                             dtype=torch.bool, device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 计算正负样本调整项
        pos_adjustment = (cos_theta - self.m_l) * self.s - self.bias
        neg_adjustment = cos_theta * self.s - self.bias
        
        # 数值稳定化处理
        pos_logits = pos_adjustment.clamp(min=-self.s, max=self.s)
        neg_logits = neg_adjustment.clamp(min=-self.s, max=self.s)
        
        # 损失计算
        pos_loss = torch.log1p(torch.exp(-pos_logits))  # log(1 + exp(-x))
        neg_loss = torch.log1p(torch.exp(neg_logits)) * self.l
        
        # 组合损失
        loss_matrix = torch.where(one_hot, pos_loss, neg_loss)
        total_loss = loss_matrix.sum(dim=1).mean()
        
        return cos_theta.detach(), total_loss

    def update_class_count(self, new_class_count):
        """
        动态更新类别样本统计（需在训练过程中同步调用）
        
        Args:
            new_class_count (dict): 新类别样本字典
        """
        self.class_count_dict = {str(k): v for k, v in new_class_count.items()}
        self._precompute_margins()

class Unified_Cross_Entropy_Loss(nn.Module):
    # 构造函数
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, r=1.0):
        # 调用父类的构造函数
        super(Unified_Cross_Entropy_Loss, self).__init__()
        # 初始化输入特征的数量
        self.in_features = in_features
        # 初始化输出特征的数量
        self.out_features = out_features
        # 初始化边界值m
        self.m = m
        # 初始化缩放因子s
        self.s = s
        # 初始化正样本损失权重
        self.l = l
        # 初始化类别重平衡因子
        self.r = r
        # 初始化偏置参数
        self.bias = Parameter(torch.FloatTensor(1))
        # 设置偏置参数的初始值
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        # 初始化权重参数
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # 设置权重参数的初始值
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

    # 前向传播函数
    def forward(self, input, label=None):
        # 计算输入和部分权重的归一化余弦相似度
        cos_theta = F.linear(input, F.normalize(self.weight, eps=1e-5))
        if label == None:
            return (cos_theta,)
        else:
            # 计算正类的修改后的余弦值
            cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
            # 计算负类的余弦值
            cos_m_theta_n = self.s * cos_theta - self.bias
            # 计算正类的损失
            p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
            # 计算负类的损失并乘以正样本损失权重
            n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l 
            # 将标签转换为one-hot编码
            one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
            one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # 计算最终损失
            loss = one_hot * p_loss + (~one_hot) * n_loss
            loss = loss.sum(dim=1).mean()
            # 返回平均损失
            return (cos_theta,loss)

class Arcface_Head(nn.Module):
    def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5):
        super(Arcface_Head, self).__init__()

        # 设置 ArcFace 头部的参数
        self.s = s
        self.m = m
        
        # 初始化权重参数，使用 Xavier 均匀初始化
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # 预计算在前向传播中使用的一些常数
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 对输入进行线性变换，使用归一化后的权重
        cosine = F.linear(input, F.normalize(self.weight))
        if label == None:
            return cosine
        else:
            # 从余弦值计算正弦值
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            
            # 基于余弦值、正弦值和预计算的常数计算最终的 phi 值
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

            # 为提供的标签创建一个独热编码张量
            one_hot = torch.zeros(cosine.size()).type_as(phi).long()
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            # 根据独热编码组合 phi 和 cosine 计算最终的输出
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

            # 通过预定义的 's' 参数缩放输出
            output *= self.s
            loss = nn.NLLLoss()(F.log_softmax(output, -1), label)
            
            return (output,loss.mean())

class SubCenterArcFaceHead(nn.Module):
    def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5, k=4):
        """
        embedding_size: 特征嵌入的维度
        num_classes: 类别数
        s: 缩放因子
        m: 边界
        k: 每个类别的中心数
        """
        super(SubCenterArcFaceHead, self).__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.k = k
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * k, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 归一化权重和特征
        cosine = F.linear(input, F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.k)
        # 选择最接近的中心
        cosine, _ = torch.max(cosine, dim=2)
        
        if label==None:
            return cosine
            
        else:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)


            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            loss = nn.NLLLoss()(F.log_softmax(output, -1), label)
            
            return (output,loss.mean())

class SubcenterArcMarginProduct(nn.Module):
    r"""Modified implementation from https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
        """
    def __init__(self, in_features, out_features, K=4, s=30.0, m=0.50, easy_margin=False):
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K
        self.weight = Parameter(torch.FloatTensor(out_features*self.K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(input, F.normalize(self.weight))
        
        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        #cos(phi+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class SubCenterArcFaceDynamicMarginHead(nn.Module):
    def __init__(self, class_count_dict={},embedding_size=128, num_classes=10575, s=64., k=4, lambda_=1/4, a=0.5, b=0.05):
        """
        embedding_size: 特征嵌入的维度
        num_classes: 类别数
        s: 缩放因子
        m: 边界
        k: 每个类别的中心数
        a,b,lambda_: 边际函数参数
        """
        super(SubCenterArcFaceDynamicMarginHead, self).__init__()
        self.num_classes = num_classes
        self.class_count_dict = class_count_dict
        self.s = s
        self.a = a
        self.b = b
        self.lambda_ = lambda_
        self.k = k
        self.weight = nn.Parameter(torch.FloatTensor(self.num_classes * k, embedding_size))
        nn.init.xavier_uniform_(self.weight)


    def _get_dynamic_margin(self, classname):
        """
        为每个类别动态计算边际
        """
        class_count = self.class_count_dict[classname]
        dynamic_margin = self.a * pow(class_count, -self.lambda_) + self.b
        return dynamic_margin

    def forward(self, input, label=None):
        cosine = F.linear(input, F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.k)
        cosine, _ = torch.max(cosine, dim=2)
        if label == None:
            return cosine
        else:
            sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

            m_l = []
            for la in label:
                la = str(la.item())
                d_m = self._get_dynamic_margin(la)
                m_l.append(d_m)
            m_l = torch.tensor(m_l).to(label.device)
            
            cos_dm = torch.cos(m_l)
            sin_dm = torch.sin(m_l)
            th = torch.cos(torch.pi - m_l)
            mm = torch.sin(torch.pi - m_l) * m_l
            # cos(A+B) = cosAcosB-sinAsinB
            phi = cosine * cos_dm.view(-1, 1) - sine * sin_dm.view(-1, 1)
            phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))

            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            loss = nn.NLLLoss()(F.log_softmax(output, -1), label)
            
            return (output,loss.mean())
        
class Unified_SubCenterArcFaceDynamicMarginHead(nn.Module):
    def __init__(self, class_count_dict={}, in_features=128, out_features=10575, s=64., k=4, lambda_=1/4, a=0.5, b=0.05, l=1.0, r=1.0):
        """
        embedding_size: 特征嵌入的维度
        num_classes: 类别数
        s: 缩放因子
        m: 边界
        k: 每个类别的中心数
        a,b,lambda_: 边际函数参数
        """
        super(Unified_SubCenterArcFaceDynamicMarginHead, self).__init__()
        self.class_count_dict = class_count_dict
        # 初始化输入特征的数量
        self.in_features = in_features
        # 初始化输出特征的数量
        self.out_features = out_features
        self.s = s
        self.a = a
        self.b = b
        self.lambda_ = lambda_
        self.k = k
        # 初始化正样本损失权重
        self.l = l
        # 初始化类别重平衡因子
        self.r = r
        # 初始化偏置参数
        self.bias = Parameter(torch.FloatTensor(1))
        # 设置偏置参数的初始值
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        # 初始化权重参数
        self.weight = Parameter(torch.FloatTensor(out_features*k, in_features))
        # 设置权重参数的初始值
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        # 注册一个缓冲区，用于存储权重的动量
        self.register_buffer('weight_mom', torch.zeros_like(self.weight))


    def _get_dynamic_margin(self, classname):
        """
        为每个类别动态计算边际
        """
        class_count = self.class_count_dict[classname]
        dynamic_margin = self.a * pow(class_count, -self.lambda_) + self.b
        return dynamic_margin

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))
        # cosine = F.linear(input, F.normalize(self.weight))
        cosine = cosine.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine, dim=2)
        if label == None:
            return cosine
        else:
            sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

            m_l = []
            for la in label:
                la = str(la.item())
                d_m = self._get_dynamic_margin(la)
                m_l.append(d_m)
            m_l = torch.tensor(m_l).to(label.device)
            
            cos_dm = torch.cos(m_l)
            sin_dm = torch.sin(m_l)
            th = torch.cos(m_l)
            mm = torch.sin(m_l) * m_l
            # cos(A+B) = cosAcosB-sinAsinB
            phi = cosine * cos_dm.view(-1, 1) + sine * sin_dm.view(-1, 1)
            phi = torch.where(cosine < th.view(-1, 1), phi, cosine + mm.view(-1, 1))
            # 在这里，使用 torch.where 根据条件选择新的余弦相似度。如果 cosine 大于阈值 th，
            # 则选择 phi，否则选择 cosine - mm。这个步骤可能是为了在一些条件下调整余弦相似度的值。
            # 计算正类的修改后的余弦值
            cos_m_theta_p = self.s * phi - self.bias
            # 计算负类的余弦值
            cos_m_theta_n = self.s * cosine - self.bias
            # 计算正类的损失
            p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
            # 计算负类的损失并乘以正样本损失权重
            n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l 
            
            # 将标签转换为one-hot编码
            one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
            one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # one_hot = torch.index_select(one_hot, 1, partial_index)
            # 计算最终损失
            loss = one_hot * p_loss + (~one_hot) * n_loss
            # 返回平均损失
            loss = loss.sum(dim=1).mean()
            # 返回平均损失
            return (cosine,loss.mean())
        
class AdaptiveSubCenterArcFace(nn.Module):
    def __init__(self, class_count_dict={}, embedding_size=128, num_classes=10575, s=64., k=20, lambda_=1/4, a=0.5, b=0.05):
        super(AdaptiveSubCenterArcFace, self).__init__()
        self.num_classes = num_classes
        self.class_count_dict = class_count_dict
        self.s = s
        self.a = a
        self.b = b
        self.lambda_ = lambda_
        self.k = k
        self.weight = nn.Parameter(torch.FloatTensor(self.num_classes * k, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # Initialize mutation coefficient and subclass centers
        self.prev_classwise_cv = nn.Parameter(torch.zeros((num_classes, 1)), requires_grad=False)
        self.centers = nn.Parameter(torch.ones((num_classes, 1)), requires_grad=False)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """重写  state_dict method to include prev_classwise_cv and centers."""
        state = super(AdaptiveSubCenterArcFace, self).state_dict(destination, prefix, keep_vars)
        state[prefix + 'prev_classwise_cv'] = self.prev_classwise_cv
        state[prefix + 'centers'] = self.centers
        return state

    def load_state_dict(self, state_dict, strict=True):
        """重写 load_state_dict 方法 to load prev_classwise_cv and centers."""
        prev_classwise_cv = state_dict.pop('prev_classwise_cv', None)
        centers = state_dict.pop('centers', None)
        super(AdaptiveSubCenterArcFace, self).load_state_dict(state_dict, strict)
        if prev_classwise_cv is not None:
            self.prev_classwise_cv.copy_(prev_classwise_cv)
        if centers is not None:
            self.centers.copy_(centers)

    def _get_dynamic_margin(self, classname):
        """
        为每个类别动态计算边际
        """
        class_count = self.class_count_dict[classname]
        dynamic_margin = self.a * pow(class_count, -self.lambda_) + self.b
        return dynamic_margin
    
    def update_classwise_coefficient_of_variation(self,batch_features, batch_labels, prev_classwise_cv, num_classes, alpha=0.1):
        """
        更新每一类的变异系数。

        参数:
            batch_features (torch.Tensor): 批次特征向量，形状为 (B, 128)
            batch_labels (torch.Tensor): 批次标签向量，形状为 (B, 1)
            prev_classwise_cv (torch.Tensor): 前一步每一类的变异系数，形状为 (num_classes, 1)
            num_classes (int): 类别总数
            alpha (float): 指数平均的衰减因子

        返回:
            torch.Tensor: 更新后的每一类变异系数，形状为 (num_classes, 1)
        """
        # 检查CUDA是否可用
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 确保所有的数据在设备上
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        prev_classwise_cv = prev_classwise_cv.to(device)

        # 初始化新的变异系数为零
        new_classwise_cv = nn.Parameter(torch.zeros(num_classes, 1, device=device))
        # print(1,type(new_classwise_cv)) 

        for class_id in range(num_classes):
            # 选取属于当前类别的特征
            class_features = batch_features[batch_labels.squeeze() == class_id]

            # 如果当前类别有数据，则计算变异系数
            if class_features.size(0) > 0:
                # 计算每个特征的均值和标准差
                mean, std = torch.mean(class_features), torch.std(class_features)
                # 再次计所有特征的平均均值和平均标准差
                avg_mean = torch.mean(mean)
                avg_std = torch.mean(std)
                # 避免除以零
                cv = avg_std / avg_mean if mean != 0 else 0
                # 更新变异系数
                new_classwise_cv = new_classwise_cv.clone()
                new_classwise_cv[class_id] =  cv 
            
            nonzero_indices = new_classwise_cv.nonzero()
            if len(nonzero_indices) > 0:
                min_val = new_classwise_cv[nonzero_indices].min()
            else:
                # 处理所有元素都是零的情况，你可以根据需要设置一个默认值或采取其他适当的措施。
                min_val = 0  # 或者你认为合适的默认值
            max_val = new_classwise_cv.max()
            normalized_classwise_cv = (new_classwise_cv - min_val) / (max_val - min_val)
            classwise_cv = alpha * normalized_classwise_cv +  (1 - alpha) * prev_classwise_cv
            classwise_cv = nn.Parameter(classwise_cv, requires_grad=False)
            
        
        return classwise_cv

    def map_coefficients_to_discrete_range(self,coefficients, k, exponent=2):
        """
        将变异系数映射到 (1, k] 的离散区间中，并确保结果为整数。

        参数:
            coefficients (torch.Tensor): 变异系数张量，形状为 (num_classes, 1)
            k (int): 映射的最大值
            exponent (float): 映射函数的指数

        返回:
            torch.Tensor: 映射后的整数张量，形状为 (num_classes, 1)
        """
        # 应用映射函数 k * x ** exponent，并四舍五入到最接近的整数
        mapped_values = torch.round(k * torch.pow(coefficients, exponent))

        # 确保值在 1 到 k 之间
        mapped_values.clamp_(min=1, max=k)
        mapped_values = nn.Parameter(mapped_values, requires_grad=False)

        return mapped_values

    def max_values_by_class_gpu(self, input_tensor, n_values_tensor):
        B, num_class, k = input_tensor.shape

        # 检查CUDA是否可用，如果可用，将张量移动到GPU
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            n_values_tensor = n_values_tensor.cuda()

        # 创建一个大小为 num_class x k 的掩码张量，并在 GPU 上初始化为 0
        mask = torch.zeros((num_class, k), device=input_tensor.device, dtype=torch.bool)

        # 填充掩码张量，使每个类的前 n 个元素为 True
        for class_idx in range(num_class):
            n_value = int(n_values_tensor[class_idx, 0].item())
            mask[class_idx, :n_value] = True
        # print(mask)
        # 扩展掩码以匹配输入张量的形状
        mask = mask.unsqueeze(0).expand(B, num_class, k)

        # 应用掩码，并将未选中的元素设置为负无穷，使用in-place操作
        input_tensor.masked_fill_(~mask, float('-inf'))

        # 沿第三维找到最大值，使用in-place操作
        result_tensor = torch.max(input_tensor, dim=2, keepdim=True).values.squeeze()
        # print(result_tensor.shape)
        return result_tensor

    def forward(self, input, label=None):
        cosine = F.linear(input, F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.k)
       
        if label == None:
            cosine = self.max_values_by_class_gpu(cosine,self.centers)
            # print(cosine.shape)
            return cosine
        else:
            # 更新类变异系数
            self.prev_classwise_cv = self.update_classwise_coefficient_of_variation(input, label,self.prev_classwise_cv,self.num_classes, alpha=0.2)
            # 根据更新的类变异系数更新每个类的子中心数量
            self.centers = self.map_coefficients_to_discrete_range(self.prev_classwise_cv,self.k)
            # 计算当前每个实例样本与本类的多个类中心最近的余弦值
            cosine = self.max_values_by_class_gpu(cosine,self.centers)
            sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

            m_l = []
            for la in label:
                la = str(la.item())
                d_m = self._get_dynamic_margin(la)
                m_l.append(d_m)
            m_l = torch.tensor(m_l).to(label.device)
            
            cos_dm = torch.cos(m_l)
            sin_dm = torch.sin(m_l)
            th = torch.cos(torch.pi - m_l)
            mm = torch.sin(torch.pi - m_l) * m_l
            # cos(A+B) = cosAcosB-sinAsinB
            phi = cosine * cos_dm.view(-1, 1) - sine * sin_dm.view(-1, 1)
            phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))

            one_hot = torch.zeros(cosine.size(), device=input.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            loss = nn.NLLLoss()(F.log_softmax(output, -1), label)
            
            return (output,loss.mean())

class Normalized_Softmax_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64, r=1.0):
        super(Normalized_Softmax_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.r = r
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.register_buffer('weight_mom', torch.zeros_like(self.weight))

    def forward(self, input, label=None):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))
        if label == None:
            return cos_theta
        else:
            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
            one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # one_hot = torch.index_select(one_hot, 1, partial_index)

            d_theta = one_hot.to(cos_theta) * self.m
            logits = self.s * (cos_theta - d_theta)
            loss = F.cross_entropy(logits, label)
            return (logits,loss)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + ', r=' + str(self.r) + ')'

class Unified_Cross_Entropy_Loss(nn.Module):
    # 构造函数
    def __init__(self, in_features, out_features, m=0.4, s=64, l=1.0, r=1.0):
        # 调用父类的构造函数
        super(Unified_Cross_Entropy_Loss, self).__init__()
        # 初始化输入特征的数量
        self.in_features = in_features
        # 初始化输出特征的数量
        self.out_features = out_features
        # 初始化边界值m
        self.m = m
        # 初始化缩放因子s
        self.s = s
        # 初始化正样本损失权重
        self.l = l
        # 初始化类别重平衡因子
        self.r = r
        # 初始化偏置参数
        self.bias = Parameter(torch.FloatTensor(1))
        # 设置偏置参数的初始值
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        # 初始化权重参数
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # 设置权重参数的初始值
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

    # 前向传播函数
    def forward(self, input, label=None):
        # 计算输入和部分权重的归一化余弦相似度
        cos_theta = F.linear(input, F.normalize(self.weight, eps=1e-5))
        if label == None:
            return (cos_theta,)
        else:
            # 计算正类的修改后的余弦值
            cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
            # 计算负类的余弦值
            cos_m_theta_n = self.s * cos_theta - self.bias
            # 计算正类的损失
            p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
            # 计算负类的损失并乘以正样本损失权重
            n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l 
            # 将标签转换为one-hot编码
            one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
            one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # 计算最终损失
            loss = one_hot * p_loss + (~one_hot) * n_loss
            loss = loss.sum(dim=1).mean()
            # 返回平均损失
            return (cos_theta,loss.mean())
        

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + ', l=' + str(self.l) \
               + ', r=' + str(self.r) + ')'

class DynamicMargin_Unified_Cross_Entropy_Loss(nn.Module):
    # 构造函数
    def __init__(self, class_count_dict={},in_features=128, out_features=10575, m=0.4, s=64, lambda_=1/4, a=0.5, b=0.05, l=1.0, r=1.0):
        # 调用父类的构造函数
        super(DynamicMargin_Unified_Cross_Entropy_Loss, self).__init__()
        self.class_count_dict = class_count_dict
        # 初始化输入特征的数量
        self.in_features = in_features
        # 初始化输出特征的数量
        self.out_features = out_features
        # 初始化边界值m
        self.m = m
        # 初始化缩放因子s
        self.s = s
        # 初始化正样本损失权重
        self.l = l
        # 初始化类别重平衡因子
        self.r = r
        self.a = a
        self.b = b
        self.lambda_ = lambda_
        # 初始化偏置参数
        self.bias = Parameter(torch.FloatTensor(1))
        # 设置偏置参数的初始值
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        # 初始化权重参数
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # 设置权重参数的初始值
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
    
    def _get_dynamic_margin(self, classname):
        """
        为每个类别动态计算边际
        """
        class_count = self.class_count_dict[classname]
        dynamic_margin = self.a * pow(class_count, -self.lambda_) + self.b
        return dynamic_margin

    # 前向传播函数
    def forward(self, input, label=None):
        # 计算输入和部分权重的归一化余弦相似度
        cos_theta = F.linear(input, F.normalize(self.weight, eps=1e-5))
        if label == None:
            return (cos_theta,)
        else:
            m_l = []
            for la in range(self.out_features):
                la = str(la)
                d_m = self._get_dynamic_margin(la)
                m_l.append(d_m)
            m_l = torch.tensor(m_l).to(label.device)

            # 计算正类的修改后的余弦值
            cos_m_theta_p = self.s * (cos_theta - m_l) - self.bias
            # 计算负类的余弦值
            cos_m_theta_n = self.s * cos_theta - self.bias
            # 计算正类的损失
            p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
            # 计算负类的损失并乘以正样本损失权重
            n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l 
            # 将标签转换为one-hot编码
            one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
            one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # 计算最终损失
            loss = one_hot * p_loss + (~one_hot) * n_loss
            loss = loss.sum(dim=1).mean()
            # 返回平均损失
            return (cos_theta,loss.mean())
        

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) \
               + ', l=' + str(self.l) \
               + ', r=' + str(self.r) + ')'

