
# 分别使用GPU和CPU测试模型速度 使用Throughput(images/sec)和Latency (ms)两个统计量
import os
import json
import torch
import sys
import time
from tqdm import tqdm
import torch.cuda as cuda
from my_dataset import MyDataSet_1
from torch.utils.data import DataLoader
# from models.FLENet import FLENet_XT,FLENet_T0,FLENet_T1,FLENet_T2
from models.CNNs.fasternet import FasterNet_T0,FasterNet_T1,FasterNet_T2
from models.Multi_FLENet import Multi_FLENet_T0
from experiments.Ensemble_learning.utils import read_data
from models.Shift.shiftvit import ShiftViT_XT,ShiftViT_T0,ShiftViT_T1,ShiftViT_T2
from models.FLENet import FLENet_T0,FLENet_T1,FLENet_T2
from models.FLENet_M0 import FLENet_M0
from models.CNNs.mobilenet_v3 import mobilenet_v3_small , mobilenet_v3_large
from models.CNNs.ghostnetv2 import ghostnetv2
from models.CNNs.mbv2_ca import MBV2_CA
# from models.FLENetV2 import FLENet_T0



# 读取数据集
def read_data(test_dir):
    # 标签转换
    json_path = r'D:\wangzhaojiang\FLENet\metrics\GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    class_indict = {value: key for key, value in class_indict.items()}

    # 初始化数组
    test_images = []
    test_labels = []


    # 遍历test子文件夹并获取其下的图像路径以及对应的标签值
    for subfolder in os.listdir(test_dir):
        subfolder_path = os.path.join(test_dir, subfolder)
        if subfolder_path.startswith('.'):
            continue
        if os.path.isdir(subfolder_path):
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)
                if image_path.startswith('.'):
                    continue
                test_images.append(image_path)
                # 标签转换
                test_labels.append(int(class_indict[subfolder]))
                # print(type(class_indict[subfolder]))
    
    print("{} images for testing.".format(len(test_images)))

    return  test_images, test_labels
# 数据集加载

# 评估

def testing(test_path):
    # 加载并准备好模型和输入数据
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FLENet_M0(num_classes=8105).to(device)
    model_weight_path = r"D:\wangzhaojiang\FLENet\experiments\Model_architecture\results\weights\FLENet_M0\best_val_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device),strict=False)
    # model.to(device)
    model.eval()
    
    test_images, test_labels = read_data(test_path)
    batch_size= 256
    # 实例化验证数据集
    test_dataset = MyDataSet_1(images_paths=test_images,
                            images_labels=test_labels,
                            img_size=224,
                            is_training=False)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=8,
                            collate_fn=test_dataset.collate_fn)

    # 创建CUDA事件
    # start_event = cuda.Event()
    # end_event = cuda.Event()

    # 计算吞吐量
    total_samples = 0
    total_time = 0

    data_loader = tqdm(test_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, paths = data
        start_time = time.time()
        # logitH, logitM, logitT = model(images.to(device))
        logitT = model(images.to(device))
        end_time = time.time()
    
        total_time += end_time - start_time
        total_samples += batch_size
    

    print(total_samples,total_time)
    throughput = total_samples / total_time  # 计算吞吐量（样本/秒）
    print("Throughput: {:.2f} samples/sec".format(throughput))

if __name__ == '__main__':
    test_path = r"C:\Users\wzj\Desktop\ks_mobilevit\datasets\test\test_1"
    testing(test_path)