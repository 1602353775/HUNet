import os
import json
import torch
import time
import sys
import gc
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from my_dataset import MyDataSet_1

from models.FLENet import FLENet_T0,FLENet_32
from models.FLENet_BN import FLENet_T1,FLENet_T2
from models.FLENet_M0 import FLENet_M0
from models.CNNs.fasternet import FasterNet_T0
from models.FasterNet_CA import FasterNet_T0_CA
from models.Shift.shiftvit import ShiftViT_T0
from models.CNNs.mobilenet_v2 import MobileNetV2
from models.CNNs.mbv2_ca import MBV2_CA
from models.CNNs.mobilenet_v3 import mobilenet_v3_large,mobilenet_v3_small
from models.CNNs.ghostnetv2 import ghostnetv2
from models.Transformer.efficientvit.efficientvit import EfficientViT
from models.CNNs.shufflenet_v2 import shufflenet_v2_x1_0,shufflenet_v2_x1_5
from models.CNNs.RepViT import repvit_m1




from PIL import Image
def check_and_delete_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                Image.open(file_path)
            except (IOError, OSError):
                print(f"Unable to open image: {file_path}")
                os.remove(file_path)


def reed_data(test_dir):
    json_path = '../GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    class_indict = {value: key for key, value in class_indict.items()}
    
    test_images_path = []
    test_images_label = []
    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        if image_path.startswith('.'):
            continue
        zi = image_name.split('_')[2]
        if zi in class_indict:
            test_images_path.append(image_path)
            # 标签转换

            test_images_label.append(int(class_indict[zi]))
        else:
            print(image_path)

    return test_images_path, test_images_label


model_zoo = {
            # 'FLENet_32': FLENet_32(num_classes=8105),
            # 'Shufflenet_v2_x1_0': shufflenet_v2_x1_0(),
            # 'shufflenet_v2_x1_5': shufflenet_v2_x1_5(),
            # 'mobilenet_v2_1.0x': MobileNetV2(num_classes=8105),
            # 'FLENet_M0':FLENet_M0(num_classes=8105),
            # 'FLENet_T0_24':FLENet_T0(num_classes=8105),
            # # 'FLENet_T0_32':FLENet_T0(num_classes=8105),
            # 'FLENet_T1':FLENet_T1(num_classes=8105),
            'FLENet_T2':FLENet_T2(num_classes=8105),
            # 'ShiftViT_T0':ShiftViT_T0(),
            # 'FasterNet_T0':FasterNet_T0(),
            # 'FasterNet_CA_T0_24':FasterNet_T0_CA(),
            # 'MBV2_CA':MBV2_CA(),
            # 'mobilenet_v3_large':mobilenet_v3_large(num_classes=8105),
            # 'mobilenet_v3_small':mobilenet_v3_small(num_classes=8105),
            # 'ghostnetv2':ghostnetv2(),
            # 'EfficientViT':EfficientViT(num_classes=8105)，
            # 'RepViT_M1':repvit_m1(pretrained=False,num_classes=8105)
            }

def main_test():
    json_path = '../GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)
        
    # batch_size = 256
    image_size = [128, 224, 224, 224, 128, 128, 128, 128, 128, 128, 224, 224, 224, 128, 224, 224]
    batch_size = [128, 256, 256, 128, 256, 256, 256, 200, 256, 256, 64, 64, 256, 256, 128, 1024]
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    memory_format = [torch.channels_last, torch.contiguous_format][1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('CUDA is available.')
        # 打印PyTorch的CUDA版本
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDA version: {torch.version.cuda}')
    else:
        print('CUDA is not available.')
    test_root = "dataset/testing_set"
    
    test_images_path,test_images_label = reed_data(test_root)
    
    model_index = 0
    for key, value in model_zoo.items():
        torch.cuda.empty_cache()
        gc.collect()
        print("开始在测试集：MACT testing 上测试模型：{},测试集样本量为：{}".format(key,len(test_images_label)))
        # 实例化验证数据集
        test_dataset = MyDataSet_1(images_paths=test_images_path,
                                    images_labels=test_images_label,
                                    img_size=image_size[model_index],
                                    is_training=False,
                                    dataset = "mact")
        
        test_loader = DataLoader(test_dataset,
                        batch_size=batch_size[model_index],
                        shuffle=False,
                        pin_memory=True,
                        num_workers=8,
                        collate_fn=test_dataset.collate_fn)
        
        weigths_path = os.path.join('../results/weights', key, 'best_val_model.pth')
        model = value.to(device=device, memory_format=memory_format)
        if os.path.exists(weigths_path):
            weights_dict = torch.load(weigths_path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                    if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
            model.eval()
        
            accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
            sample_num = 0
            total_time = 0
            data_loader = tqdm(test_loader, file=sys.stdout)
            
            results_df = pd.DataFrame(columns=['Image Path', 'Label', 'Prediction', 'Correct Prediction'])
                
            for step, data in enumerate(data_loader):
                images, labels, paths = data
                sample_num += images.shape[0]

                start_time = time.time()
                pred = model(images.to(device))
                end_time = time.time()
                total_time += end_time - start_time
                pred_classes = torch.max(pred, dim=1)[1]
                correct_preds = torch.eq(pred_classes, labels.to(device))
                accu_num += torch.eq(pred_classes, labels.to(device)).sum()
                throughput = sample_num / total_time
                
                # 更新data_loader的描述信息
                data_loader.desc = "[model: {}] datasets: MACR, acc: {:.5f},Throughput-{}: {:.2f} samples/sec ".format(
                    key, accu_num.item() / sample_num, device, throughput)

                # 将数值标签转换为具体类别标签
                labels = [class_indict[str(label.item())] for label in labels]
                pred_classes = [class_indict[str(pred.item())] for pred in pred_classes]

                # 将这一批次的结果添加到DataFrame中
                batch_results = pd.DataFrame({
                    'Image Path': paths,
                    'Label': labels,
                    'Prediction': pred_classes,
                    'Correct Prediction': correct_preds.int().tolist()  # 将正确性结果转换为整数列表
                })
                results_df = pd.concat([results_df, batch_results], ignore_index=True)
            
            # 在DataFrame末尾添加总结行
            summary_data = {
                'Image Path': 'Summary',
                'Label': None,
                'Prediction': None,
                'Correct Prediction': 'Total Samples: {}, Correct Predictions: {}'.format(sample_num, accu_num.item())
            }
            summary_df = pd.DataFrame([summary_data])
            results_df = pd.concat([results_df, summary_df], ignore_index=True)

            # 计算总体正确率并追加到DataFrame
            overall_accuracy = (accu_num.item() / sample_num) * 100
            accuracy_data = {
                'Image Path': 'Overall Accuracy',
                'Label': None,
                'Prediction': None,
                'Correct Prediction': '{:.2f}%'.format(overall_accuracy)
            }
            accuracy_df = pd.DataFrame([accuracy_data])
            results_df = pd.concat([results_df, accuracy_df], ignore_index=True)

            # 确保结果目录存在
            results_dir = '../results'
            os.makedirs(results_dir, exist_ok=True)

            # 将DataFrame写入Excel文件
            results_path = os.path.join(results_dir, 'MACR_test_results.xlsx')
            results_df.to_excel(results_path, index=False)
            
            model_index+=1
                                                                              
                                                

if __name__ == '__main__':
    # folder_path = r"D:\wangzhaojiang\书法真迹\test\test_5"
    # check_and_delete_images(folder_path)
    main_test()