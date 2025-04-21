import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from models.HUNet import HUNet_24
from collections import defaultdict

def resize_and_pad(img):
    """调整图像尺寸并填充到128x128，使用边缘像素填充（优化版）"""
    original_width, original_height = img.size
    max_dim = max(original_width, original_height)
    
    # 调整图像尺寸（保持长宽比）
    if max_dim > 128:
        ratio = 128 / max_dim
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        img = img.resize((new_width, new_height), Image.BILINEAR)
    else:
        new_width, new_height = original_width, original_height
    
    # 创建目标图像并直接填充边缘像素
    padded_img = Image.new("RGB", (128, 128))
    
    # 计算粘贴位置（居中）
    x_offset = (128 - new_width) // 2
    y_offset = (128 - new_height) // 2
    
    # 填充边缘像素（优化实现）
    if new_width < 128:
        # 水平填充（左右边缘扩展）
        left_col = img.crop((0, 0, 1, new_height))
        right_col = img.crop((new_width-1, 0, new_width, new_height))
        
        # 左侧填充
        for x in range(x_offset):
            padded_img.paste(left_col, (x, y_offset))
        
        # 右侧填充
        for x in range(x_offset + new_width, 128):
            padded_img.paste(right_col, (x, y_offset))
    
    if new_height < 128:
        # 垂直填充（上下边缘扩展）
        top_row = img.crop((0, 0, new_width, 1))
        bottom_row = img.crop((0, new_height-1, new_width, new_height))
        
        # 上方填充
        for y in range(y_offset):
            padded_img.paste(top_row, (x_offset, y))
        
        # 下方填充
        for y in range(y_offset + new_height, 128):
            padded_img.paste(bottom_row, (x_offset, y))
    
    # 粘贴原始图像（居中）
    padded_img.paste(img, (x_offset, y_offset))
    
    return padded_img

class EVOBCCharactersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = resize_and_pad(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]),
#     'test': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
# }

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.128, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.128, 0.225])
    ])
}


# 修改数据集路径
data_dir = r'D:\古文字识别数据集\数据集\EVOBC\EVOBC'
model_path = 'experiments/EVOBC_best_model.pth'
full_dataset = EVOBCCharactersDataset(root_dir=data_dir, transform=data_transforms['train'])

# 新的数据划分逻辑
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(full_dataset.samples):
    class_indices[label].append(idx)

train_indices = []
test_indices = []

for label, indices in class_indices.items():
    if len(indices) <= 10:
        train_indices.extend(indices)
        print(f"Class {label} has {len(indices)} samples (<=10), all added to training set")
    else:
        X_dummy = np.zeros(len(indices))
        y_dummy = np.array([label] * len(indices))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_idx, test_idx in sss.split(X_dummy, y_dummy):
            train_indices.extend([indices[i] for i in train_idx])
            test_indices.extend([indices[i] for i in test_idx])
        print(f"Class {label} has {len(indices)} samples (>10), split into {len(train_idx)} train and {len(test_idx)} test")

# 创建数据集子集
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

# 设置数据集属性
train_dataset.classes = full_dataset.classes
test_dataset.classes = full_dataset.classes

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HUNet_24(num_classes=len(full_dataset.classes)).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.01, total_iters=10),
        CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-7)
    ],
    milestones=[10]
)

def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    num_classes = len(full_dataset.classes)


    # 如果已有加载模型，初始化best_acc
    if os.path.exists(model_path):
        with torch.no_grad():
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, num_classes)
            best_acc = test_acc
            print(f"[初始测试准确率] Loss: {test_loss:.4f} Macro Acc: {best_acc:.4f}")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_bar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}', ncols=100)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs[0], 1)
            loss = criterion(outputs[0], labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{torch.sum(preds == labels.data).item()/inputs.size(0):.2f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        scheduler.step()
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 使用Macro-Averaged Accuracy进行评估
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, num_classes)
        print(f'Test Loss: {test_loss:.4f} Macro Acc: {test_acc:.4f}\n')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'experiments/Lontar_best_model.pth')
            print(f'*** New best model saved (Macro Acc: {best_acc:.4f}) ***\n')
    
    print(f'Best Test Macro Acc: {best_acc:.4f}')
    return model

def evaluate_model(model, dataloader, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    
    eval_bar = tqdm(dataloader, desc='Evaluating', ncols=100)
    with torch.no_grad():
        for inputs, labels in eval_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs[0], 1)
            loss = criterion(outputs[0], labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # 统计每个类别的正确预测
            correct = preds == labels
            for c in range(num_classes):
                class_mask = (labels == c)
                class_correct[c] += (correct & class_mask).sum()
                class_total[c] += class_mask.sum()
            
            eval_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct.sum().item()/labels.size(0):.2f}'
            })
    
    total_loss = running_loss / len(dataloader.dataset)
    
    # 计算Macro-Averaged Accuracy
    class_acc = torch.zeros(num_classes, device=device)
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc[c] = class_correct[c].float() / class_total[c].float()
        else:
            class_acc[c] = 0.0
    macro_avg_acc = class_acc.mean().item()
    
    return total_loss, macro_avg_acc


if os.path.exists(model_path):
    print(f"\n{'='*40}")
    print(f"发现预训练模型：{model_path}")
    print("正在加载模型参数...")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")
        print(f"{'='*40}\n")
        
        # 验证加载的模型
        with torch.no_grad():
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, len(full_dataset.classes))
            print(f"已加载模型的测试准确率：{test_acc:.4f}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        print("将从头开始训练...\n")
else:
    print("\n未找到预训练模型，将进行初始化训练...\n")



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    num_epochs = 100
    model = train_model(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs
    )

    final_loss, final_acc = evaluate_model(model, test_loader, criterion, len(full_dataset.classes))
    print(f'Final Test Macro Accuracy: {final_acc:.4f}')