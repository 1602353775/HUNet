import os
import gc
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# 自定义模块
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate, load_dataset
from datasets_count import count_images
from models.HUNet import HUNet_24, HUNet_32, HUNet_36, HUNet_48

# 常量定义
CHECKPOINT_EXTN = "pt"
IMG_SIZES = {  # 不同尺寸配置
    "s": [64, 64], 
    "m": [128, 128],
    "l": [256, 256],
    "n": [224, 224]
}

def setup_environment(args):
    """初始化训练环境，包括设备检测、目录创建等"""
    # 自动选择训练设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 打印CUDA信息
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'PyTorch版本: {torch.__version__}')
        print(f'CUDA版本: {torch.version.cuda}')
    else:
        print('使用CPU进行训练')
    
    # 创建必要目录
    os.makedirs("./experiments/weights", exist_ok=True)
    os.makedirs("./experiments/runs", exist_ok=True)
    os.makedirs("./experiments/best_val_checkpoint", exist_ok=True)
    os.makedirs("./experiments/val_Error statistics", exist_ok=True)
    
    return device

def create_data_loaders(args, input_size):
    """
    创建训练和验证数据加载器
    参数:
        args - 命令行参数
        input_size - 输入图像尺寸
    返回:
        train_loader, val_loader - 数据加载器
    """
    # 加载数据集
    train_images, train_labels, test_images, test_labels = load_dataset(args.data_path)
    
    # 实例化数据集
    train_dataset = MyDataSet(
        image_paths=train_images,
        image_labels=train_labels,
        img_size=input_size[0],
        is_training=True
    )
    
    val_dataset = MyDataSet(
        image_paths=test_images,
        image_labels=test_labels,
        img_size=input_size[0],
        image_type="rgb",
        is_training=False
    )
    
    # 计算workers数量
    nw = min(os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 6)
    print(f'使用{nw}个数据加载进程')
    
    # 创建带类别平衡的采样器
    # sampler = get_sampler()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=sampler(train_dataset, num_samples_cls=args.num_classes),
        pin_memory=True,
        num_workers=nw,
        persistent_workers=True if nw > 0 else False,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=True if nw > 0 else False,
        collate_fn=val_dataset.collate_fn
    )
    
    return train_loader, val_loader

def initialize_model(model_name, device, args):
    """
    初始化模型和优化器，支持从检查点恢复完整训练状态
    参数:
        model_name - 模型名称
        device - 训练设备
        args - 命令行参数
    返回:
        model - 初始化后的模型
        optimizer - 优化器（包含恢复的状态）
        start_epoch - 起始epoch
        best_val_acc - 最佳验证准确率
    """
    # 模型定义
    model_zoo = {
        'HUNet_24': HUNet_24(args.num_classes),
        'HUNet_32': HUNet_32(args.num_classes),
        'HUNet_36': HUNet_36(args.num_classes),
        'HUNet_48': HUNet_48(args.num_classes),
    }
    model = model_zoo[model_name].to(device)
    
    # 初始化训练状态
    best_val_acc = 0
    start_epoch = 0
    optimizer = None
    
    # 优先加载检查点（包含完整训练状态）
    if args.checkpoint:
        ckpt_path = os.path.join(args.checkpoint, model_name, 'best_val_checkpoint.pt')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            # 加载模型状态
            # 直接非严格模式加载（快速方案）
            # model.load_state_dict(model_state_dict, strict=False)

            model_state_dict = checkpoint['model_state_dict']
            
            try:
                # 先尝试严格匹配加载
                model.load_state_dict(model_state_dict)
                print("严格模式加载成功")
            except RuntimeError as e:
                print(f"检测到键不匹配，启动自动过滤：{str(e)}")
                # 获取当前模型状态键集合
                current_model_keys = set(model.state_dict().keys())
                # 过滤检查点中不存在的键
                filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                      if k in current_model_keys}
                # 加载过滤后的状态字典
                model.load_state_dict(filtered_state_dict, strict=False)
                # 打印被忽略的键
                ignored_keys = set(model_state_dict.keys()) - current_model_keys
                print(f"已忽略 {len(ignored_keys)} 个不匹配键：{list(ignored_keys)[:3]}...")

            # 初始化优化器后再加载状态
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                params, 
                lr=args.lr, 
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            if 'optimizer_state_dict' in checkpoint:
                try:
                    # 获取当前模型参数映射
                    model_params = {id(p): p for p in model.parameters() if p.requires_grad}
                    
                    # 过滤优化器状态
                    filtered_optimizer_state = {
                        'param_groups': checkpoint['optimizer_state_dict']['param_groups'],
                        'state': {}
                    }
                    
                    # 重建状态字典
                    for p_id, state in checkpoint['optimizer_state_dict']['state'].items():
                        # 查找当前参数是否仍然存在
                        current_param = model_params.get(p_id)
                        if current_param is not None and current_param.shape == state['exp_avg'].shape:
                            filtered_optimizer_state['state'][p_id] = state
                        else:
                            print(f"忽略过期参数状态: {p_id}")

                    # 加载过滤后的优化器状态
                    optimizer.load_state_dict(filtered_optimizer_state)
                    print("优化器状态已安全恢复")
                except Exception as e:
                    print(f"优化器状态恢复失败: {str(e)}")
                    print("将重新初始化优化器...")
                    # 重新初始化优化器参数组
                    params = [p for p in model.parameters() if p.requires_grad]
                    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
            
            # 恢复训练状态
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint["best_metric"]
            print(f'成功恢复检查点: epoch={start_epoch}, val_acc={best_val_acc:.2f}')
            
            # 直接返回，检查点优先级最高
            return model, optimizer, start_epoch, best_val_acc
    
    # 加载预训练权重（当无检查点时）
    if args.weights:
        weights_path = os.path.join(args.weights, model_name)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f'成功加载预训练权重: {weights_path}')
    
    # 冻结非head层
    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad_(False)
            else:
                print(f'可训练层: {name}')
    
    # 初始化优化器（无检查点时）
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params, 
        lr=args.lr, 
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    return model, optimizer, start_epoch, best_val_acc

import gc
import torch

def train_model(args, device):
    """主训练流程"""
    # 数据准备
    input_size = IMG_SIZES[args.img_size]
    train_loader, val_loader = create_data_loaders(args, input_size)
    
    # 模型集合
    models_to_train = ['HUNet_24', 'HUNet_32', 'HUNet_36', 'HUNet_48']
    
    for model_name in models_to_train:
        print(f'\n{"="*60} 训练模型 {model_name} {"="*60}')
        
        # 模型初始化
        model, optimizer, start_epoch, best_val_acc = initialize_model(model_name, device, args)
        
        # 学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # TensorBoard记录器
        tb_writer = SummaryWriter(f"./experiments/runs/{model_name}/")
        
        # 训练循环
        for epoch in range(start_epoch, args.epochs):
            # 训练阶段
            train_loss, train_acc = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
                num_classes=args.num_classes
            )
            
            # 更新学习率
            scheduler.step()
            
            # 评估前的清理工作
            torch.cuda.empty_cache()  # 清理PyTorch的CUDA缓存
            gc.collect()  # 执行Python垃圾回收
            
            # 验证阶段
            excel_dir = f"./experiments/val_Error statistics/{model_name}/"
            os.makedirs(excel_dir, exist_ok=True)
            val_loss, val_acc = evaluate(
                model=model,
                model_name=model_name,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                excel_path=os.path.join(excel_dir, f"{epoch}.xlsx"),
                num_classes=args.num_classes,
                count=(best_val_acc > 0.95)  # 当准确率>90%时统计错误
            )
            
            # 记录训练指标
            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("train_acc", train_acc, epoch)
            tb_writer.add_scalar("val_loss", val_loss, epoch)
            tb_writer.add_scalar("val_acc", val_acc, epoch)
            tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_dir = f"./experiments/weights/{model_name}/"
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存模型权重
                torch.save(model.state_dict(), os.path.join(save_dir, "best_val_model.pth"))
                
                # 保存检查点
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    "best_metric": best_val_acc,
                    "model_name": model_name,
                }
                torch.save(checkpoint, f"./experiments/best_val_checkpoint/{model_name}/best_val_checkpoint.pt")
                
                print(f"Epoch {epoch}: 验证准确率提升至 {best_val_acc:.2%}")

            # 每个epoch结束后的清理
            torch.cuda.empty_cache()
            gc.collect()

def main(args):
    """主函数"""
    if args.datasets_count:
        count_images(args.data_path, "./datasets.xlsx")
    
    device = setup_environment(args)
    print(device)
    print("\n训练配置:")
    print(f"- 图像尺寸: {IMG_SIZES[args.img_size]}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 初始学习率: {args.lr}")
    print(f"- 训练轮次: {args.epochs}\n")
    
    train_model(args, device)

if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="HUNet训练脚本")
    parser.add_argument('--img_size', type=str, default="m", choices=['s', 'm', 'l', 'n'])
    parser.add_argument('--num_classes', type=int, default=8105)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-path', type=str, default='./datasets')
    parser.add_argument('--datasets_count', action='store_true', default=True)
    parser.add_argument('--weights', type=str, default="")
    parser.add_argument('--checkpoint', type=str, default="./experiments/best_val_checkpoint/")
    parser.add_argument('--freeze-layers', action='store_true')
    parser.add_argument('--device', default='cuda', help='训练设备(cuda/cpu)')
    
    args = parser.parse_args()
    
    # 启动训练
    main(args)