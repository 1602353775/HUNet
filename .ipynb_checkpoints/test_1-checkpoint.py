import os
import json
import torch
import time
import sys
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from my_dataset import MyDataSet_1

from models.FLENet import FLENet_T0, FLENet_32
from models.FLENet_BN import FLENet_T1, FLENet_T2
from models.FLENet_M0 import FLENet_M0
# from models.FLENetV2 import 
from models.CNNs.fasternet import FasterNet_T0
from models.FasterNet_CA import FasterNet_T0_CA
from models.Shift.shiftvit import ShiftViT_T0
from models.CNNs.mobilenet_v2 import MobileNetV2
from models.CNNs.mbv2_ca import MBV2_CA
from models.CNNs.mobilenet_v3 import mobilenet_v3_large, mobilenet_v3_small
from models.CNNs.ghostnetv2 import ghostnetv2
from models.CNNs.shufflenet_v2 import shufflenet_v2_x1_0, shufflenet_v2_x1_5
from models.Transformer.efficientvit.efficientvit import EfficientViT
from models.FLENet_XR import FLENet_T0_all_eca
from models.FLENetV2 import FLENet_V2_T0
from models.FLENet_MixStyle import FLENet_T0_mixstyle
from models.FLENet_24_all_ECA import FLENet_T0_all_eca
from models.FLENet_24_no_ECA import FLENet_T0_no_eca
from models.FLENet_24_local_ECA import FLENet_T0_local_eca
from models.FLENet_24_global_ECA import FLENet_T0_global_eca

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
    json_path = './GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    class_indict = {value: key for key, value in class_indict.items()}

    test_images_path = []
    test_images_label = []
    for subfolder in os.listdir(test_dir):
        subfolder_path = os.path.join(test_dir, subfolder)
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

    return test_images_path, test_images_label


model_zoo = {
    # 'FLENet_24_2284':FLENet_T0(num_classes=8105),
    # 'FLENet_24_2284':FLENet_T0(num_classes=8105),
    # 'FLENet_T0_all_eca':FLENet_T0_all_eca(num_classes=8105),
    # 'FLENet_T0_no_eca':FLENet_T0_no_eca(num_classes=8105),
    # 'FLENet_T0_local_eca':FLENet_T0_local_eca(num_classes=8105),
    # 'FLENet_T0_global_eca':FLENet_T0_global_eca(num_classes=8105),
    # 'FLENet_T0_mixstyle_v2':FLENet_T0_mixstyle(num_classes=8105),
    # 'FLENet_24_4462':FLENet_T0(num_classes=8105),
    # 'FLENet_32': FLENet_32(num_classes=8105),
    # 'FLENet_T0_global_eca':FLENet_T0_all_eca(num_classes=8105),  # all ECA
    # 'shufflenet_v2_x1_0': shufflenet_v2_x1_0(),
    # 'shufflenet_v2_x1_5': shufflenet_v2_x1_5(),
    # 'mobilenet_v2_1.0x': MobileNetV2(num_classes=8105),
    # 'FLENet_M0': FLENet_M0(num_classes=8105),
    # 'FLENet_T0_24': FLENet_T0(num_classes=8105),
    # 'FLENet_T1': FLENet_T1(num_classes=8105),
    'FLENet_T2': FLENet_T2(num_classes=8105),
    # 'ShiftViT_T0': ShiftViT_T0(),
    # 'FasterNet_T0': FasterNet_T0(),
    # 'FasterNet_CA_T0_24': FasterNet_T0_CA(),
    # 'MBV2_CA': MBV2_CA(),
    # 'mobilenet_v3_large_extra': mobilenet_v3_large(num_classes=8105),
    # 'mobilenet_v3_small': mobilenet_v3_small(num_classes=8105),
    # 'ghostnetv2': ghostnetv2(),
    # 'EfficientViT': EfficientViT(num_classes=8105)8105
}


def pth_to_onnx():
    image_size = [128,128,128,128,128, 224, 224, 224, 128, 128, 128, 128, 128, 128, 224, 224, 224, 128, 224, 224]
    batch_size = [256,256,256,256,256, 256, 256, 256, 256, 256, 256, 200, 256, 256, 64, 64, 256, 256, 128, 1024]

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    memory_format = [torch.channels_last, torch.contiguous_format][1]

    dir = 'experiments/Model_architecture/results/onnx'
    model_index = 0
    for key, value in model_zoo.items():
        weigths_path = os.path.join('./results/weights', key, 'best_val_model.pth')
        model = value.to(device=device, memory_format=memory_format)
        if os.path.exists(weigths_path):
            weights_dict = torch.load(weigths_path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
            model.eval()
            # 定义输入和输出张量的名称和形状
            input_names = ["input"]
            output_names = ["output"]
            batch_size = 1
            input_shape = (batch_size, 3, image_size[model_index], image_size[model_index])
            output_shape = (batch_size, 8105)

            name = key + '.onnx'
            path = os.path.join(dir, name)
            # 将 PyTorch 模型转换为 ONNX 格式
            torch.onnx.export(
                model,  # 要转换的 PyTorch 模型
                torch.randn(input_shape),  # 模型输入的随机张量
                path,  # 保存的 ONNX 模型的文件名
                input_names=input_names,  # 输入张量的名称
                output_names=output_names,  # 输出张量的名称
                dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}
                # 动态轴，即输入和输出张量可以具有不同的批次大小
            )
            model_index += 1


def Convert_ONNX():
    image_size = [128,128, 128, 128, 128, 128, 224, 224, 224, 128, 128, 128, 128, 128, 128, 224, 224, 224, 128, 224, 224]
    batch_size = [128,256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 200, 256, 256, 64, 64, 256, 256, 128, 1024]
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    memory_format = [torch.channels_last, torch.contiguous_format][1]

    dir = r'D:\wangzhaojiang\FLENet\experiments\Model_architecture\results\onnx'
    model_index = 0
    for key, value in model_zoo.items():
        print(key)
        weigths_path = os.path.join('./results/weights', key, 'best_val_model.pth')
        model = value.to(device=device, memory_format=memory_format)
        if os.path.exists(weigths_path):
            weights_dict = torch.load(weigths_path, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
            model.eval()

            name = key + '.onnx'
            path = os.path.join(dir,name)
            # Let's create a dummy input tensor
            batch_size = 1
            input_shape = (batch_size, 3, image_size[model_index], image_size[model_index])
            dummy_input = torch.randn(input_shape, requires_grad=True).to(device)

            # Export the model   
            torch.onnx.export(model,  # model being run
                              dummy_input,  # model input (or a tuple for multiple inputs)
                              path,  # where to save the model
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=14,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['modelInput'],  # the model's input names
                              output_names=['modelOutput'],  # the model's output names
                              dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                            'modelOutput': {0: 'batch_size'}},
                              keep_initializers_as_inputs=False)
            print('Model {} has been converted to ONNX'.format(key))
            model_index += 1


def main_test():
    # batch_size = 256
    image_size = [128,128, 128, 128, 128, 128, 224, 224, 224, 128, 128, 128, 128, 128, 128, 224, 224, 224, 128, 224, 224]
    batch_size = [128,256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 200, 256, 256, 64, 64, 256, 256, 128, 1024]
    # batch_size = [256, 256, 256, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    memory_format = [torch.channels_last, torch.contiguous_format][1]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('CUDA is available.')
        # 打印PyTorch的CUDA版本
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDA version: {torch.version.cuda}')
    else:
        print('CUDA is not available.')
    test_root = r"D:\wangzhaojiang\书法真迹\test"

    for i in range(1, 6):  # 在5个测试集上测试
        if i == 1:
            test_n = 'test_' + str(1)
            test_dir = r"C:\Users\wzj\Desktop\ks_mobilevit\datasets\test\test_1"
        else:
            test_n = 'test_' + str(i)
            test_dir = os.path.join(test_root, test_n)
        print(test_dir)
        test_images_path, test_images_label = reed_data(test_dir)

        model_index = 0
        for key, value in model_zoo.items():
            torch.cuda.empty_cache()
            gc.collect()
            print("开始在测试集：{} 上测试模型：{},测试集样本量为：{}".format(test_n, key, len(test_images_label)))
            # 实例化验证数据集
            test_dataset = MyDataSet_1(images_paths=test_images_path,
                                       images_labels=test_images_label,
                                       img_size=image_size[model_index],
                                       is_training=False)

            test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size[model_index], # batch_size[model_index]
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=8,
                                     collate_fn=test_dataset.collate_fn)

            weigths_path = os.path.join('./results/weights', key, 'best_val_model.pth')
            model = value.to(device=device, memory_format=memory_format)
            if os.path.exists(weigths_path):
                weights_dict = torch.load(weigths_path, map_location=device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(load_weights_dict, strict=True)
                model.eval()

                accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
                sample_num = 0
                total_time = 0
                data_loader = tqdm(test_loader, file=sys.stdout)
                for step, data in enumerate(data_loader):
                    images, labels, paths = data
                    sample_num += images.shape[0]

                    start_time = time.time()
                    pred = model(images.to(device))
                    end_time = time.time()
                    total_time += end_time - start_time
                    pred_classes = torch.max(pred, dim=1)[1]
                    accu_num += torch.eq(pred_classes, labels.to(device)).sum()
                    throughput = sample_num / total_time

                    data_loader.desc = "[model: {}] datasets: {}, acc: {:.5f},Throughput-{}: {:.2f} samples/sec ".format(
                        key, test_n, accu_num.item() / sample_num, device, throughput)
                    # if step == 5000:
                    #     break
            model_index += 1


if __name__ == '__main__':
    # folder_path = r"D:\wangzhaojiang\书法真迹\test\test_5"
    # check_and_delete_images(folder_path)
     main_test()
    #  Convert_ONNX()
