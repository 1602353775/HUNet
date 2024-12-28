from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import cv2
import os 
import csv
import os
import json
import torch
from PIL import Image
from torchvision import transforms as T

# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")


def read_excel_to_list(file_path):
    # 读取excel文件
    df = pd.read_excel(file_path)
    
    # 第二列的标题
    second_column_title = df.columns[0]
    
    # 获取第二列的前50个数据
    data_list = df[second_column_title].head(50).tolist()
    
    return data_list

# 定义一个函数，将图片路径、真实标签、预测标签、特征写入CSV文件
def write_to_csv(image_path, true_label, predicted_label, features,csv_file):
    
    # 判断CSV文件是否已存在，如果不存在，则先写入表头
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Path', 'Label', 'Predicted', 'Features'])
    
    # 将图片路径、真实标签、预测标签、特征写入CSV文件
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_path, true_label, predicted_label, features.tolist()])

def pre_process(image):
    img_size = 128
    img = Image.open(image).convert("RGB")
    width, height = img.size
    if height>width:
        new_width = int(img_size*(width/height))
        img = img.resize((new_width,img_size),resample=Image.Resampling.BILINEAR)
        l_padding = int((img_size-new_width)/2)
        r_padding = int(img_size-l_padding-new_width)
        padding = (l_padding,0,r_padding,0)
    else:
        new_height = int(img_size*(height/width))
        img = img.resize((img_size, new_height),resample=Image.Resampling.BILINEAR)
        t_padding = int((img_size - new_height) / 2)
        b_padding = int(img_size - t_padding - new_height)
        padding = (0,t_padding,0,b_padding)

    transforms = T.Compose([T.Pad(padding, fill=255, padding_mode='edge'),  # constant edge reflect symmetric
                                T.ToTensor(),
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])
    img = transforms(img)
    img = torch.unsqueeze(img, dim=0).to(device)
    return img



from models.FLENet import FLENet_T0
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
model= FLENet_T0(num_classes=8105)
weigths_path = r"D:\wangzhaojiang\FLENet\experiments\Model_architecture\results\weights\FLENet_T0_24\best_val_model.pth"
weights_dict = torch.load(weigths_path, map_location=device)
load_weights_dict = {k: v for k, v in weights_dict.items()
                        if model.state_dict()[k].numel() == v.numel()}
model.load_state_dict(load_weights_dict, strict=False)
model = model.eval().to(device)

file_path = r"D:\wangzhaojiang\FLENet\experiments\Ensemble_learning\datasets_train.xlsx"

HanZi = read_excel_to_list(file_path)

test_root = r'D:\wangzhaojiang\书法真迹\test'

json_path = r'D:\wangzhaojiang\FLENet\GF_class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as f:
    class_indict = json.load(f)


for zi in HanZi:
    print(zi)
    images = []
    file_name = 'top-100.csv'
    csv_file = os.path.join(r'D:\wangzhaojiang\FLENet\features',file_name)
    print(csv_file)
    
    test_n = os.listdir(test_root)
    for subset in test_n:
        dir = os.path.join(test_root,subset,zi)
        if os.path.exists(dir):
            names = os.listdir(dir)
            for name in names:
                path = os.path.join(dir,name)
                images.append(path)

    for img in images:
        image = pre_process(img)
        logits,feats = model(image)
        output = logits.squeeze().detach().cpu()
        probs = torch.softmax(output, dim=0)
        
        predict_cla = torch.argmax(probs).numpy()
        predicted_label = class_indict[str(predict_cla)]
        features = feats.squeeze().detach().cpu().numpy()
        write_to_csv(img, zi, predicted_label, features,csv_file)

        
    