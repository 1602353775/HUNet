import os
import gc
import argparse
import torch
import warnings
import collections
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from my_dataset import MyDataSet_1,MyDataSet_2
from data.sampers.ClassAwareSampler import get_sampler
from experiments.Model_architecture.utils import train_one_epoch, evaluate_one_epoch,read_data_1,evaluate_one_epoch_onnx
from data.sampers.EffectNumSampler import *

from datasets_count import count_images

from models.FLENet import FLENet_T0,FLENet_T1,FLENet_T2,FLENet_32
from models.FLENet_M0 import FLENet_M0
from models.FLENet_MixStyle import FLENet_T0_mixstyle
from models.FLENet_XR import FLENet_T0_all_eca
from models.FLENet_GCL import FLENet_GCL_XT,FLENet_GCL_T0,FLENet_GCL_T1

from models.RIDE_FLENet import RIDE_FLENet_T0,RIDE_FLENet_T1,RIDE_FLENet_T2
from models.EA_FLENet import EA_FLENet_T0,EA_FLENet_T1,EA_FLENet_T2
from models.Multi_FLENet import Multi_FLENet_T0,Multi_FLENet_T1,Multi_FLENet_T2
from models.CNNs.fasternet import FasterNet_T0,FasterNet_T1,FasterNet_T2
from models.Shift.shiftvit import ShiftViT_XT,ShiftViT_T0,ShiftViT_T1,ShiftViT_T2

from models.CNNs.ghostnetv2 import ghostnetv2
from models.CNNs.mbv2_ca import MBV2_CA
from models.CNNs.mobilenet_v3 import mobilenet_v3_small , mobilenet_v3_large
from models.FasterNet_CA import FasterNet_T0
from models.CNNs.ghostnet import ghostnet
from models.CNNs.densenet import densenet121
from models.CNNs.resnet import resnet18,resnet34,resnet50
from models.CNNs.shufflenetv2 import ShuffleNetV2
from models.CNNs.mobilenet_v2 import MobileNetV2
from models.CNNs.mobilenet_v3 import MobileNetV3
from models.CNNs.mbv2_ca import MBV2_CA
from models.CNNs.moganet import MogaNet
from models.Transformer.efficientvit.efficientvit import EfficientViT
from models.Hybrid.mobilevit_v1.model import mobile_vit_xx_small,mobile_vit_x_small,mobile_vit_small
from models.Hybrid.efficientformer_v2 import efficientformerv2_s1
from models.CNNs.RepViT import repvit_m1


from read_sfzd import sfzd
from read_ygsf import ygsf



CHECKPOINT_EXTN = "pt"

def main(args):
    if args.datasets_count:
        samples_per_cls = count_images(args.train_path, "./datasets_train.xlsx")
        test_samples_per_cls = count_images(args.test_path, "./datasets_test.xlsx")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('CUDA is available.')
        # 打印PyTorch的CUDA版本
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDA version: {torch.version.cuda}')
    else:
        print('CUDA is not available.')

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    if os.path.exists("./results/weights") is False:
        os.makedirs("./results/weights")
    if os.path.exists("./results/best_val_checkpoint") is False:
        os.makedirs("./results/best_val_checkpoint")
    if os.path.exists("./results/val_Error statistics") is False:
        os.makedirs("./results/val_Error statistics")
    

    # 数据读取
    if args.Multi_head:
        train_images, train_labels, test_images, test_labels = read_data_1(args.train_path,args.test_path)

        img_size = {"s": [64, 64], "m": [128, 128], "l": [256, 256], "n": [224, 224]}

        input_size = img_size[args.img_size]
        

    else:
        data_dir = r"C:\Users\wzj\Desktop\ks_mobilevit\datasets\extra"

        train_images, train_labels, test_images, test_labels = read_data_1(args.train_path,args.test_path)
        extra_train_images, extra_train_labels = ygsf (data_dir)
        print(len(extra_train_labels))
        

        img_size = {"s": [64, 64], "m": [128, 128], "l": [256, 256], "n": [224, 224]}

        input_size = img_size[args.img_size]
        # 实例化训练数据集
        binary_train_dataset = MyDataSet_1(images_paths=train_images,
                                images_labels=train_labels,
                                img_size=input_size[0],
                                img_type='Binary_image',
                                is_training=True)

        # 实例化训练数据集
        extra_train_dataset = MyDataSet_1(images_paths=extra_train_images,
                                    images_labels= extra_train_labels,
                                    img_size=input_size[0],
                                    img_type='RGB_image',
                                    is_training=True)
        train_dataset = ConcatDataset([binary_train_dataset,extra_train_dataset])

        # print("全部训练集样本量为：{}".format(len(train_dataset)))
        # 实例化验证数据集
        val_dataset = MyDataSet_1(images_paths=test_images,
                                images_labels=test_labels,
                                img_size=input_size[0],
                                is_training=False)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
     
    # 采样器选择
    if args.train_rule == 'None':
        train_sampler = None  
        per_cls_weights = None 
    elif args.train_rule == 'BalancedRS':
        train_sampler = BalancedDatasetSampler(train_dataset)
        per_cls_weights = None
    elif args.train_rule == 'EffectNumRS':
        train_sampler = EffectNumSampler(train_dataset)
        per_cls_weights = None
    elif args.train_rule == 'CBENRS':
        train_sampler = CBEffectNumSampler(train_dataset)
        per_cls_weights = None  
    elif args.train_rule == 'ClassAware':
        train_sampler = ClassAwareSampler(train_dataset)
        per_cls_weights = None
    elif args.train_rule == 'EffectNumRW':
        train_sampler = None
        sampler = EffectNumSampler(train_dataset)
        per_cls_weights = sampler.per_cls_weights/sampler.per_cls_weights.sum()  
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    elif args.train_rule == 'BalancedRW':
        train_sampler = None
        sampler = BalancedDatasetSampler(train_dataset)
        per_cls_weights = sampler.per_cls_weights/sampler.per_cls_weights.sum()   
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)        
    else:
        warnings.warn('Sample rule is not listed')

    
    train_loader = DataLoader(binary_train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=binary_train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn)

    # memory format
    memory_format = [torch.channels_last, torch.contiguous_format][1]

    model_zoo = {
                 'RepViT_M1':repvit_m1(num_classes=args.num_classes),
                'efficientformerv2_s1':efficientformerv2_s1(),
                'FLENet_32':FLENet_32(num_classes=args.num_classes),
                 'FLENet_24_4462': FLENet_T1(num_classes=args.num_classes),
                 'FLENet_24_4484':FLENet_T0(num_classes=args.num_classes),
                'FLENet_T2':FLENet_T2(num_classes=args.num_classes),
                'FLENet_M0':FLENet_M0(num_classes=args.num_classes),
                'mobilenet_v3_large_extra':mobilenet_v3_large(num_classes=args.num_classes),
                'mobilenet_v3_small':mobilenet_v3_small(num_classes=args.num_classes),
                'FLENet_T0_24':FLENet_T0(num_classes=args.num_classes),
                'FLENet_T0_all_eca':FLENet_T0_all_eca(num_classes=args.num_classes),
                'FLENet_T0_mixstyle_v2':FLENet_T0_mixstyle(num_classes=args.num_classes),
                'FasterNet_T0':FasterNet_T0()
                'FasterNet_CA_T0_24':FasterNet_T0(),
                'MBV2_CA':MBV2_CA()
                'ghostnetv2':ghostnetv2()
                'ShiftViT_T0':ShiftViT_T0(),
                 'ShiftViT_XT_2': ShiftViT_XT(),
                 'ShiftViT_T1':ShiftViT_T1(),
                 'FLENet_LDAM_XT':FLENet_T0(num_classes=args.num_classes),
                 'FLENet_IB_XT':FLENet_T0(num_classes=args.num_classes),
                 'FLENet_GCL_XT_SGD':FLENet_GCL_XT(num_classes=args.num_classes),
                 'FLENet_GCL_XT':FLENet_GCL_XT(num_classes=args.num_classes),
                 'FLENet_T0_24':FLENet_T0(num_classes=args.num_classes),
                 'FLENet_T1':FLENet_T1(num_classes=args.num_classes),
                 'FLENet_T2':FLENet_T2(num_classes=args.num_classes),
                 }

    for key, value in model_zoo.items():
        best_train_acc = 0
        best_val_acc = 0
        count = False
        start_epoch = 0
        model_name = key
        model = value.to(device=device, memory_format=memory_format)
        print("===========================================模型  {}  开始训练！==============================================".format(model_name))

        if args.weights != "":
            # best_val_weights_path = os.path.join(args.weights, model_name, 'best_val_model.pth')
            best_val_weights_path = args.weights
            if os.path.exists(best_val_weights_path):
                weights_dict = torch.load(best_val_weights_path, map_location=device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))
            else:
                raise FileNotFoundError("not found checkpoints file: {}".format(args.checkpoints))

        # 是否冻结权重
        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

        pg = [p for p in model.parameters() if p.requires_grad]

        # optimizer = optim.SGD(pg,
        #     lr=0.1,                 # 初始学习率
        #     momentum=0.9,           # 动量
        #     weight_decay=0.0005     # 权重衰减（L2正则化）
        # )
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[5, 10, 15],  # 在第100和150个epoch时降低学习率
        #     gamma=0.1               # 每次降低前学习率以0.1倍衰减
        # )
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        # LambdaLR更新学习率方式是 lr = lr*lr_lambda 其中，lr由optim系列优化器提供，lr_lambda由lr_scheduler>lambdaLR提供
        # scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=lf)
        # optimizer = optim.AdamW(pg,lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)  # 最新优化
        # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        # 定义优化器
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.025)
        # 定义学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)


        if args.checkpoint != "":
            best_val_checkpoint_path = os.path.join(args.checkpoint, model_name, 'best_val_checkpoint.pt')
            if os.path.exists(best_val_checkpoint_path):
                print(best_val_checkpoint_path)
                checkpoint = torch.load(best_val_checkpoint_path, map_location=device)
                start_epoch = checkpoint["epoch"]+1
                best_metric = checkpoint["best_metric"]
                best_val_acc = best_metric
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer.state = collections.defaultdict(dict, optimizer.state)  # 解决从 CPU 加载到 GPU 的兼容问题
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                model_name = checkpoint["model_name"]


                print("--------------成功加载 {} 的checkpoint， val_acc：{}，  epoch：{}！------------".format(model_name,best_val_acc,start_epoch))
            else:
                print("---------------------暂时没有{}的checkpoint，需要从头开始训练！------------------".format(model_name))
        

        

        for epoch in range(start_epoch, args.epochs):

            onnx_acc = evaluate_one_epoch_onnx(model_path = r"D:\wangzhaojiang\FLENet\mobilenet_v3_large.onnx",data_loader=val_loader)
            print(onnx_acc)
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    samples_per_cls=[],
                                                    loss_type = args.loss_type,
                                                    loss_fn = args.loss_fn,
                                                    no_of_classes=args.num_classes,
                                                    )

            scheduler.step()
            #
            if best_val_acc > 0.85:
                count = True

            # validate
            if os.path.exists("./results/val_Error statistics/{}".format(model_name)) is False:
                os.makedirs("./results/val_Error statistics/{}".format(model_name))
            excel_path = "./results/val_Error statistics/{}/{}.xlsx".format(model_name, epoch)
            val_loss, val_acc = evaluate_one_epoch(model=model,
                                         model_name=model_name,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch,
                                         excel_path=excel_path,
                                         num_classes=args.num_classes,
                                         count=count)
            log_dir = "./results/runs/{}/".format(model_name)
            tb_writer = SummaryWriter(log_dir)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                if os.path.exists("./results/weights/{}".format(model_name)) is False:
                    os.makedirs("./results/weights/{}".format(model_name))
                torch.save(model.state_dict(), "./results/weights/{}/best_train_model.pth".format(model_name))
            if val_acc > best_val_acc:
                print("正确率上升！")
                best_val_acc = val_acc
                if os.path.exists("./results/weights/{}".format(model_name)) is False:
                    os.makedirs("./results/weights/{}".format(model_name))
                torch.save(model.state_dict(), "./results/weights/{}/best_val_model.pth".format(model_name))

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    "best_metric": val_acc,
                    "model_name": model_name,
                }
                if os.path.exists("./results/best_val_checkpoint/{}".format(model_name)) is False:
                    os.makedirs("./results/best_val_checkpoint/{}".format(model_name))

                ckpt_name = "./results/best_val_checkpoint/{}/best_val_checkpoint.{}".format(model_name, CHECKPOINT_EXTN)
                torch.save(checkpoint, ckpt_name) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=str, default="n")
    parser.add_argument('--num_classes', type=int, default=8105)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss_type', type=str, default='softmax')
    parser.add_argument('--loss_fn', type=str, default='CE')
    parser.add_argument('--train_rule', type=str, default='None')
    parser.add_argument('--re_sample', type=bool, default=False)
    parser.add_argument('--re_weight', type=bool, default=False)
    parser.add_argument('--Multi_head', type=bool, default=False,help='分类头数量，默认使用单个分类头')

    # 数据集所在根目录
    parser.add_argument('--train_path', type=str, default=r"C:\Users\wzj\Desktop\ks_mobilevit\datasets\train")
    parser.add_argument('--test_path', type=str, default=r"C:\Users\wzj\Desktop\ks_mobilevit\datasets\test")
    parser.add_argument('--datasets_count', type=bool, default=False)
    parser.add_argument('--weights', type=str, default=r"", help='initial weights path') # ./results/weights   D:\wangzhaojiang\FLENet\experiments\Model_architecture\results\weights\mobilenet_v3_large\best_val_model.pth
    parser.add_argument('--checkpoint', type=str, default="./results/best_val_checkpoint", help='continue training from checkpoint')  # ./results/best_val_checkpoint
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
