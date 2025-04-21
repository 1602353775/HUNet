import os
import sys
import json
import pickle
import random
import torch
import torch.nn.functional as F
from typing import Union,Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
from openpyxl import Workbook, load_workbook


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def save_wrong_predictions_to_excel(image_paths, true_labels, predicted_labels, excel_path, 
                                  class_indices=None, header_added=False):
    """批量保存错误预测到Excel的优化版本

    Args:
        image_paths: 图像路径列表 (List[str])
        true_labels: 真实标签Tensor (torch.Tensor)
        predicted_labels: 预测标签Tensor (torch.Tensor)
        excel_path: Excel保存路径 (str)
        class_indices: 类别字典 (dict, optional)
        header_added: 是否已存在表头 (bool)
    """
    # 转换为numpy数组提升处理速度
    true_np = true_labels.cpu().numpy() if hasattr(true_labels, 'cpu') else true_labels
    pred_np = predicted_labels.cpu().numpy() if hasattr(predicted_labels, 'cpu') else predicted_labels
    
    # 向量化筛选错误样本
    mask = true_np != pred_np
    wrong_paths = [image_paths[i] for i in np.where(mask)[0]]
    wrong_true = true_np[mask]
    wrong_pred = pred_np[mask]

    # 转换标签为类别名称
    if class_indices:
        wrong_true_names = [class_indices[str(int(label))] for label in wrong_true]
        wrong_pred_names = [class_indices[str(int(label))] for label in wrong_pred]
    else:
        wrong_true_names = wrong_true.astype(str)
        wrong_pred_names = wrong_pred.astype(str)

    # 构建DataFrame
    df = pd.DataFrame({
        "Image Path": wrong_paths,
        "True Label": wrong_true_names,
        "Predicted Label": wrong_pred_names
    })

    # 使用高效写入模式
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            startrow = writer.sheets['Sheet1'].max_row
            df.to_excel(writer, index=False, header=(not header_added), 
                      startrow=startrow if header_added else 0)
    else:
        df.to_excel(excel_path, index=False, header=(not header_added))


def write_to_xlsx(tags, correct_nums, total_nums, file_path):
    """
    优化版类别准确率统计保存函数
    
    参数说明：
    tags: 类别ID到中文标签的映射字典 (dict)
    correct_nums: 各类别正确预测数 (torch.Tensor)
    total_nums: 各类别总样本数 (torch.Tensor)
    file_path: 输出文件路径 (str)
    """
    # ================== 预处理优化 ==================
    # 统一设备处理 (支持CPU/GPU自动转换)
    device = correct_nums.device
    total_nums = total_nums.to(device)
    
    # 向量化计算 (避免逐元素操作)
    with torch.no_grad():
        # 处理除零情况并保留原始数据类型
        accuracies = torch.where(
            total_nums > 0,
            (correct_nums / total_nums) * 100.0,
            torch.zeros_like(correct_nums, dtype=torch.float32)
        )
        # 使用原生四舍五入方法
        accuracies_rounded = torch.round(accuracies * 10000) / 10000

    # ================== 数据转换优化 ==================
    # 批量转换到CPU (减少设备切换次数)
    cpu_correct = correct_nums.cpu()
    cpu_total = total_nums.cpu()
    cpu_acc = accuracies_rounded.cpu()

    # 使用字典推导式构建数据
    data_dict = {
        "tag": [tags[str(i)] for i in range(len(tags))],  # 保持顺序
        "Accuracy": cpu_acc.numpy(),
        "Correct Num": cpu_correct.numpy(),
        "Total Num": cpu_total.numpy()
    }

    # ================== DataFrame 优化 ==================
    # 直接创建排序后的DataFrame
    df = pd.DataFrame(data_dict).sort_values(
        by="Accuracy", 
        ascending=False,
        ignore_index=True  # 重置索引避免存储冗余数据
    )

    # ================== 存储优化 ==================
    # 使用高效Excel写入参数
    writer_kwargs = {
        "engine": "xlsxwriter",
        "options": {
            "strings_to_numbers": True,  # 优化数值存储
            "freeze_panes": (1, 0)       # 冻结首行
        }
    }
    
    # 上下文管理器写入
    with pd.ExcelWriter(file_path, **writer_kwargs) as writer:
        df.to_excel(
            writer, 
            index=False,
            sheet_name="Accuracy Report",
            header=["类别", "准确率 (%)", "正确数", "总数"]
        )
        
        # 自动调整列宽
        worksheet = writer.sheets["Accuracy Report"]
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(idx, idx, max_len)

    print(f"[Success] 各类别准确率统计已保存至 {file_path}")


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def load_dataset(dataset_path):
    datasets_dir = 'datasets'
    train_csv_path = os.path.join(datasets_dir, 'train.csv')
    test_csv_path = os.path.join(datasets_dir, 'test.csv')

   # 检查是否存在CSV文件
    if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
        train_images, train_labels = [], []
        try:
            with open(train_csv_path, 'r', encoding='utf-8') as f:  # 显式指定UTF-8编码
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    if len(row) >= 2:  # 确保行数据完整
                        train_images.append(row[0])
                        train_labels.append(int(row[1]))
        except UnicodeDecodeError:
            # 如果UTF-8解码失败，尝试备用编码
            with open(train_csv_path, 'r', encoding='gb18030', errors='replace') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        train_images.append(row[0].encode('utf-8', errors='replace').decode('utf-8'))
                        train_labels.append(int(row[1]))
        
        test_images, test_labels = [], []
        try:
            with open(test_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        test_images.append(row[0])
                        test_labels.append(int(row[1]))
        except UnicodeDecodeError:
            with open(test_csv_path, 'r', encoding='gb18030', errors='replace') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        test_images.append(row[0].encode('utf-8', errors='replace').decode('utf-8'))
                        test_labels.append(int(row[1]))
        
        print("从CSV文件加载数据集。")
        print("训练集图片数量: {}".format(len(train_images)))
        print("测试集图片数量: {}".format(len(test_images)))
        return train_images, train_labels, test_images, test_labels
    else:
        train_dir = os.path.join(dataset_path, "train")
        test_dir = os.path.join(dataset_path, "test")

        # 加载类别标签映射
        json_path = './GF_class_indices.json'
        with open(json_path, "r") as f:
            class_indict = json.load(f)  # 格式: {类别名: 标签}
        class_indict = {value: key for key, value in class_indict.items()}

        # 初始化数据存储列表
        train_images, train_labels = [], []
        test_images, test_labels = [], []

        # 处理训练集
        for style_folder in os.listdir(train_dir):
            style_path = os.path.join(train_dir, style_folder)
            if os.path.isdir(style_path) and not style_folder.startswith('.'):
                for char_folder in os.listdir(style_path):
                    char_path = os.path.join(style_path, char_folder)
                    if os.path.isdir(char_path) and not char_folder.startswith('.'):
                        for img_name in os.listdir(char_path):
                            img_path = os.path.join(char_path, img_name)
                            if os.path.isfile(img_path) and not img_name.startswith('.'):
                                try:
                                    train_labels.append(class_indict[char_folder])
                                    train_images.append(img_path)
                                except KeyError:
                                    print(f"警告: 跳过未知类别 '{char_folder}' 的图片: {img_path}")
                                    continue

        # 处理测试集
        for style_folder in os.listdir(test_dir):
            style_path = os.path.join(test_dir, style_folder)
            if os.path.isdir(style_path) and not style_folder.startswith('.'):
                for char_folder in os.listdir(style_path):
                    char_path = os.path.join(style_path, char_folder)
                    if os.path.isdir(char_path) and not char_folder.startswith('.'):
                        for img_name in os.listdir(char_path):
                            img_path = os.path.join(char_path, img_name)
                            if os.path.isfile(img_path) and not img_name.startswith('.'):
                                try:
                                    test_labels.append(class_indict[char_folder])
                                    test_images.append(img_path)
                                except KeyError:
                                    print(f"警告: 跳过未知类别 '{char_folder}' 的图片: {img_path}")
                                    continue

        # 创建数据集目录并保存CSV
        os.makedirs(datasets_dir, exist_ok=True)
        
        with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:  # 添加编码参数
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label'])
            for img, lbl in zip(train_images, train_labels):
                writer.writerow([str(img).encode('utf-8').decode('utf-8'),  # 确保字符串编码
                                str(lbl).encode('utf-8').decode('utf-8')])
                
        # 保存测试集CSV
        with open(test_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label'])
            for img, lbl in zip(test_images, test_labels):
                writer.writerow([str(img).encode('utf-8').decode('utf-8'),
                                str(lbl).encode('utf-8').decode('utf-8')])

        print("数据集总图片数: {}".format(len(train_images) + len(test_images)))
        print("训练集图片数: {}".format(len(train_images)))
        print("测试集图片数: {}".format(len(test_images)))
        print("已创建数据集CSV文件并保存至datasets目录。")

        return train_images, train_labels, test_images, test_labels

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def crossEntropy(softmax, logit, label, weight):
    target = F.one_hot(label,20)
    loss = - (weight * (target * torch.log(softmax(logit)+1e-7)).sum(dim=1)).sum()
    return loss


def weighted_loss(predictions,targets, num_classes):
    # 计算每个类别的数量
    label_counts = torch.bincount(targets, minlength=num_classes).float()

    # 计算每个类别的权重
    class_weights = 1 / label_counts
    class_weights /= class_weights.sum()

    # 计算加权交叉熵损失函数
    loss = F.cross_entropy(predictions, targets, weight=class_weights)

    return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing = 0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, paths = data
        sample_num += images.shape[0]

        pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # loss = weighted_loss(pred,labels.to(device),num_classes)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()  # 参考 https://blog.csdn.net/yangwangnndd/article/details/95622893
        # 梯度累加则实现了batchsize的变相扩大，如果accumulation_steps为8，则batchsize '变相' 扩大了8倍，是我们这种乞丐实验室解决显存受限的一个不错的trick，使用时需要注意，学习率也要适当放大。
        optimizer.zero_grad()  

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



@torch.no_grad()
def evaluate(model, model_name, data_loader, device, epoch, excel_path,num_classes=500,count=False):
    # acc_count = {}
    # error = []
    json_path = './GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # for index in class_indict.key():
    loss_function = torch.nn.CrossEntropyLoss()
         
    model.eval()
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
 
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels, paths = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        c = (pred_classes == labels.to(device)).squeeze()
        for i in range(len(labels.to(device))):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.5f}, acc: {:.5f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if count:
            if step == 0:
                save_wrong_predictions_to_excel(paths, labels, pred_classes, excel_path, class_indict, header_added=False)
            else:
                save_wrong_predictions_to_excel(paths, labels, pred_classes, excel_path, class_indict, header_added=True)

    if count:
        if os.path.exists("./experiments/acc_count/{}".format(model_name)) is False:
            os.makedirs("./experiments/acc_count/{}".format(model_name))

        file_name = './experiments/acc_count/{}/best_accuracy_results_{}.xlsx'.format(model_name,epoch)
        write_to_xlsx(class_indict, class_correct, class_total, file_name)


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, accumulation_steps=1):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels, paths = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_num += images.shape[0]

      # 前向传播（混合精度）
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            pred = model(images)
            loss = loss_function(pred[0], labels)
            loss = loss / accumulation_steps  # 梯度累积时的损失平均
        
        # 反向传播（自动梯度缩放）
        scaler.scale(loss).backward()

        # 梯度累积更新
        if (step + 1) % accumulation_steps == 0 or (step + 1 == len(data_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        with torch.no_grad():
            pred_classes = torch.max(pred[0], dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
            accu_loss += loss.detach() * accumulation_steps  # 恢复实际损失值

        # 更新进度条信息
        data_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def train_one_epoch_v2(model, optimizer, data_loader, device, epoch, num_classes, accumulation_steps=1):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels, paths = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_num += images.shape[0]

        # 前向传播（混合精度）
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            # 模型返回格式 (cos_theta, total_loss)
            cos_theta, loss = model(images,labels)
            loss = loss / accumulation_steps  # 梯度累积时的损失平均
        
        # 反向传播（自动梯度缩放）
        scaler.scale(loss).backward()

        # 梯度累积更新
        if (step + 1) % accumulation_steps == 0 or (step + 1 == len(data_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        # 精度计算（使用detach后的cos_theta）
        with torch.no_grad():
            pred_classes = torch.max(cos_theta, dim=1)[1]  # 直接使用cos_theta计算预测
            accu_num += torch.eq(pred_classes, labels).sum()
            accu_loss += loss.detach() * accumulation_steps  # 恢复实际损失值

        # 更新进度条信息
        data_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
    
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, model_name, data_loader, device, epoch, excel_path, num_classes=8105, count=False):
    json_path = './GF_class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    wrong_samples = [] if count else None
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels, paths = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_num += images.shape[0]

        pred = model(images)
        pred_classes = torch.max(pred[0], dim=1)[1]

        # 向量化统计类别准确率
        correct = pred_classes == labels
        accu_num += correct.sum()
        accu_loss += loss_function(pred[0], labels)

        # 使用bincount代替循环
        all_labels = labels
        correct_labels = all_labels[correct]
        class_correct += torch.bincount(correct_labels, minlength=num_classes)
        class_total += torch.bincount(all_labels, minlength=num_classes)

        # 收集错误样本信息
        if count:
            wrong_mask = ~correct
            batch_wrong_paths = [paths[i] for i in torch.where(wrong_mask)[0].cpu().numpy()]
            batch_wrong_true = labels[wrong_mask].cpu().numpy().tolist()
            batch_wrong_pred = pred_classes[wrong_mask].cpu().numpy().tolist()
            wrong_samples.extend(zip(batch_wrong_paths, batch_wrong_true, batch_wrong_pred))

        data_loader.desc = "[valid epoch {}] loss: {:.5f}, acc: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    # 批量保存错误预测
    if count and wrong_samples:
        save_wrong_predictions_to_excel(wrong_samples, excel_path, class_indict)

    # 保存类别准确率
    if count:
        os.makedirs(f"./experiments/acc_count/{model_name}", exist_ok=True)
        file_name = f'./experiments/acc_count/{model_name}/best_accuracy_results_{epoch}.xlsx'
        write_to_xlsx(class_indict, class_correct.cpu(), class_total.cpu(), file_name)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_v2(model, model_name, data_loader, device, epoch, excel_path, num_classes=8105, count=False):
    json_path = './GF_class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model.eval()
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    wrong_samples = [] if count else None
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels, paths = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_num += images.shape[0]

        # 模型返回格式 (cos_theta, loss)
        cos_theta, loss = model(images,labels)
        
        # 计算预测类别
        pred_classes = torch.max(cos_theta, dim=1)[1]
        correct = pred_classes == labels

        # 累计指标
        accu_num += correct.sum()
        accu_loss += loss  # 直接使用模型计算的loss

        # 类别级统计（向量化操作）
        class_correct += torch.bincount(labels[correct], minlength=num_classes)
        class_total += torch.bincount(labels, minlength=num_classes)

        # 错误样本收集
        if count:
            wrong_mask = ~correct
            batch_wrong_indices = torch.where(wrong_mask)[0].cpu()
            
            # 批量处理提高效率
            wrong_samples.extend([
                (paths[i], labels[i].item(), pred_classes[i].item())
                for i in batch_wrong_indices
            ])

        # 更新进度条
        data_loader.desc = "[valid epoch {}] loss: {:.5f}, acc: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    # 批量保存错误预测
    if count and wrong_samples:
        save_wrong_predictions_to_excel(wrong_samples, excel_path, class_indict)

    # 保存类别准确率
    if count:
        os.makedirs(f"./experiments/acc_count/{model_name}", exist_ok=True)
        file_name = f'./experiments/acc_count/{model_name}/best_accuracy_results_{epoch}.xlsx'
        write_to_xlsx(class_indict, class_correct.cpu(), class_total.cpu(), file_name)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def load_state_dict(model, state_dict):
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model


def load_checkpoint(resume_loc,model,optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(resume_loc,map_location=device)
    start_epoch = checkpoint["epoch"] + 1
    best_metric = checkpoint["best_metric"]
    model = load_state_dict(model, checkpoint["model_state_dict"])
    optimizer = load_state_dict(optimizer,checkpoint["optim_state_dict"])
    return (
        model,
        optimizer,
        start_epoch,
        best_metric,
    )


