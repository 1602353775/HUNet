import os
import json
import torch
from tqdm import tqdm
from typing import List, Tuple
from PIL import Image
from my_dataset import MyDataSet_1
from torch.utils.data import DataLoader
from models.FLENet_BN import FLENet_T2
import sys


def check_and_delete_image(image_path):
    try:
        with Image.open(image_path) as img:
            return True
    except:
        return False
    finally:
        os.remove(image_path)

def get_image_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                ext = os.path.splitext(file)[1]
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                    image_paths.append(file_path)
    return image_paths


def read_images(root_dir: str) -> Tuple[List[str], List[str]]:
    """
    读取多层文件夹中的所有图片，并返回图片路径和标签。

    :param root_dir: 根目录，从该目录开始递归查找图片。
    :param extensions: 需要读取的图片后缀名，默认为None，表示读取所有类型的图片。例如['.jpg', '.png']。
    :return: 图片路径和标签。
    """
    json_path = './GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    class_indict = {value: key for key, value in class_indict.items()}


    image_paths = []
    labels = []

    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if subfolder_path.startswith('.'):
            continue
        if subfolder in class_indict:
            paths = get_image_paths(subfolder_path)
            for path in paths:
                # if check_and_delete_image(path):
                image_paths.append(path)
                labels.append(int(class_indict[subfolder]))
                
    return image_paths, labels

def testing (model, data_loader, device,cuo_txt,err_txt,low_txt):
    json_path = './GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, paths = data
        sample_num += images.shape[0]
        labels = labels.to(device)
        images = images.to(device)
        preds = model(images)
        # 获取预测概率
        probs = torch.softmax(preds, dim=1)
        # 获取预测标签
        predicted_labels = preds.argmax(dim=1)

        # 找到预测
        indices_acc = torch.where((predicted_labels == labels) & (probs.max(dim=1).values < 0.6))[0]
        acc_paths = [paths[i] for i in indices_acc.tolist()]
        for i in range(len(acc_paths)):
            info = acc_paths[i]
            with open(low_txt, 'a', encoding='utf-8') as f:
                f.write(info)
                f.write('\n')

        indices_err = torch.where((predicted_labels != labels) & (probs.max(dim=1).values < 0.85))[0]
        acc_paths = [paths[i] for i in indices_err.tolist()]
        for i in range(len(acc_paths)):
            info = acc_paths[i]
            with open(cuo_txt, 'a', encoding='utf-8') as f:
                f.write(info)
                f.write('\n')
        
        # 找到预测错误但预测概率大于0.8的图片索引
        indices = torch.where((predicted_labels != labels) & (probs.max(dim=1).values >= 0.85))[0]
        # 获取预测失败图片的路径、标签和模型预测结果
        err_paths = [paths[i] for i in indices.tolist()]
        err_labels = labels[indices].tolist()
        err_preds = predicted_labels[indices].tolist()

        pred_classes = torch.max(preds, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        for i in range(len(err_paths)):
            info = err_paths[i] + '-' + class_indict[str(err_labels[i])] + '-' + class_indict[str(err_preds[i])]
            with open(err_txt, 'a', encoding='utf-8') as f:
                f.write(info)
                f.write('\n')
        
        data_loader.desc = "acc: {:.5f}".format(accu_num.item() / sample_num)

    return accu_num.item() / sample_num

def test(path,cuo_txt,err_txt,low_txt):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)
    # extensions = ['.jpg', '.png']
    image_paths, labels = read_images(path)
    # print(len(labels))
    test_dataset = MyDataSet_1(images_paths=image_paths,
                             images_labels=labels,
                             img_size=128,
                             is_training=False)
    test_loader = DataLoader(test_dataset,
                            batch_size=128,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8,
                            collate_fn=test_dataset.collate_fn)
    model = FLENet_T2(num_classes=8105).to(device=device)
    # model_weight_path = "experiments/Model_architecture/results/weights/FLENet_T2/best_val_model.pth"
    model_weight_path = r"D:\wangzhaojiang\FLENet\experiments\Model_architecture\results\weights\FLENet_T2\best_train_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    acc = testing(model, test_loader, device,cuo_txt,err_txt,low_txt)
    print(acc)
if __name__ == '__main__':
    fonts = ['cs']
    for font in fonts:
        txt_path = r"C:\Users\Administrator\Desktop\check\\"
        root = r"D:\wangzhaojiang\书法真迹\ygsf_zj\以观书法"
        path = os.path.join(root,font)
        cuo_txt = txt_path +  font + '_acc.txt'
        err_txt = txt_path + font + '_err.txt'
        low_txt = txt_path + font + '_low.txt'
        print('开始检查%s!'%font)
        test(path,cuo_txt,err_txt,low_txt)