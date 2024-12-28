import os
import gc
import argparse
import torch
import warnings
import json
import collections
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from my_dataset import MyDataSet_1,MyDataSet_2
from data.sampers.ClassAwareSampler import get_sampler
from experiments.Model_architecture.utils import train_one_epoch, evaluate_one_epoch,read_data_1
from data.sampers.EffectNumSampler import *

from datasets_count import count_images

# from models.FLENet import FLENet_T0,FLENet_T1,FLENet_T2
from models.FLENet_M0 import FLENet_M0
# from models.FLENet_GCL import FLENet_GCL_XT,FLENet_GCL_T0,FLENet_GCL_T1
#
# from models.RIDE_FLENet import RIDE_FLENet_T0,RIDE_FLENet_T1,RIDE_FLENet_T2
# from models.EA_FLENet import EA_FLENet_T0,EA_FLENet_T1,EA_FLENet_T2
# from models.Multi_FLENet import Multi_FLENet_T0,Multi_FLENet_T1,Multi_FLENet_T2
# from models.CNNs.fasternet import FasterNet_T0,FasterNet_T1,FasterNet_T2
# from models.Shift.shiftvit import ShiftViT_XT,ShiftViT_T0,ShiftViT_T1,ShiftViT_T2
#
# from models.CNNs.ghostnetv2 import ghostnetv2
# from models.CNNs.mbv2_ca import MBV2_CA
from models.CNNs.mobilenet_v3 import mobilenet_v3_small , mobilenet_v3_large
# from models.FasterNet_CA import FasterNet_T0
# from models.CNNs.ghostnet import ghostnet
# from models.CNNs.densenet import densenet121
# from models.CNNs.resnet import resnet18,resnet34,resnet50
# from models.CNNs.shufflenetv2 import ShuffleNetV2
# from models.CNNs.mobilenet_v2 import MobileNetV2
# from models.CNNs.mobilenet_v3 import MobileNetV3
# from models.CNNs.mbv2_ca import MBV2_CA
# from models.CNNs.moganet import MogaNet
# from models.Transformer.efficientvit.efficientvit import EfficientViT
# from models.Hybrid.mobilevit_v1.model import mobile_vit_xx_small,mobile_vit_x_small,mobile_vit_small




# 读取数据集
def read_data_1(train_dir,test_dir):
    train_root = train_dir
    print(train_root)
    test_root = test_dir
    print(test_root)
    # 标签转换
    json_path = './GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    class_indict = {value: key for key, value in class_indict.items()}

    train_fonts = [font for font in os.listdir(train_root) if os.path.isdir(os.path.join(train_root,font))]
    test_fonts = [font for font in os.listdir(test_root) if os.path.isdir(os.path.join(test_root,font))]
    
    train_images_path = []   # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []     # 存储验证集的所有图片路径
    test_images_label = []    # 存储验证集图片对应索引信息
    
    # 遍历每个文件夹下的文件
    for font in train_fonts:
        font_dir = os.path.join(train_root,font)
        for subfolder in os.listdir(font_dir):
            subfolder_path = os.path.join(font_dir, subfolder)
            if subfolder_path.startswith('.'):
                continue
            if os.path.isdir(subfolder_path):
                for image_name in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, image_name)
                    if image_path.startswith('.'):
                        continue
                    train_images_path.append(image_path)
                    # 标签转换
                    train_images_label.append(int(class_indict[subfolder]))
        
    for font in test_fonts:
        font_dir = os.path.join(test_root,font)
        for subfolder in os.listdir(font_dir):
            subfolder_path = os.path.join(font_dir, subfolder)
            if subfolder_path.startswith('.'):
                continue
            if os.path.isdir(subfolder_path):
                for image_name in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, image_name)
                    if image_path.startswith('.'):
                        continue
                    test_images_path.append(image_path)
                    # 标签转换
                    test_images_label.append(int(class_indict[subfolder]))



    print("{} images were found in the dataset.".format(len(train_images_path)+len(test_images_path)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(test_images_path)))

    return train_images_path, train_images_label, test_images_path, test_images_label







def test(test_path):
    tests = os.listdir(test_path)





if __name__ == '__main__':
    test_path = ''
    test(test_path)