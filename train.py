import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from my_dataset import MyDataSet_1
from utils import train_one_epoch, evaluate,load_dataset_2
from datasets_count import count_images
from model.FLENet import FLENet_T0,FLENet_T1,FLENet_T2
from ClassAwareSampler import get_sampler


CHECKPOINT_EXTN = "pt"


def main(args):
    if args.datasets_count:
        count_images(args.data_path, "./datasets.xlsx")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")

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

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    if os.path.exists("./best_val_checkpoint") is False:
        os.makedirs("./best_val_checkpoint")
    if os.path.exists("./val_Error statistics") is False:
        os.makedirs("./val_Error statistics")

    # train_images_path, train_images_label, val_images_path, val_images_label, sampler_train_images_path, sampler_train_images_label = read_split_data(
    #     args.data_path)
    train_images, train_labels, test_images, test_labels = load_dataset_2(args.data_path)

    img_size = {"s": [64, 64], "m": [128, 128], "l": [256, 256], "n": [224, 224]}

    input_size = img_size[args.img_size]

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images,
                              images_class=train_labels,
                              img_size=input_size[0],
                              is_training=True)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=test_images,
                            images_class=test_labels,
                            img_size=input_size[0],
                            is_training=False)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    sampler = get_sampler()
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=sampler(train_dataset,num_samples_cls=args.num_classes),
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn)

    # memory format
    memory_format = [torch.channels_last, torch.contiguous_format][1]

    model_zoo = {
                 'FLENet_T0':FLENet_T0(),
                 'FLENet_T1':FLENet_T1(),
                 'FLENet_T2':FLENet_T2(),
                 }

    for key, value in model_zoo.items():
        best_train_acc = 0
        best_val_acc = 0
        count = False
        start_epoch = 0
        model_name = key
        model = value.to(device=device, memory_format=memory_format)
        print("===========================================模型  {}  开始训练！================================================".format(model_name))

        if args.weights != "":
            best_val_weights_path = os.path.join(args.weights,model_name)
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

        # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
        optimizer = optim.AdamW(pg,lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)  # 最新优化
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        # LambdaLR更新学习率方式是 lr = lr*lr_lambda 其中，lr由optim系列优化器提供，lr_lambda由lr_scheduler>lambdaLR提供
        # scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda=lf)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        # torch.utils.checkpoint.checkpoint(model, input, retain_graph=capturable)
        if args.checkpoint != "":
            best_val_checkpoint_path = os.path.join(args.checkpoint, model_name, 'best_val_checkpoint.pt')
            print(best_val_checkpoint_path)
            if os.path.exists(best_val_checkpoint_path):
                checkpoint = torch.load(best_val_checkpoint_path, map_location=device)
                start_epoch = checkpoint["epoch"]+1
                best_metric = checkpoint["best_metric"]
                best_val_acc = best_metric
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                model_name = checkpoint["model_name"]

                print("--------------成功加载 {} 的checkpoint， val_acc：{}，  epoch：{}！------------".format(model_name,best_val_acc,start_epoch))
            else:
                print("--------------暂时没有{}的checkpoint，需要从头开始训练！------------".format(model_name))


        for epoch in range(start_epoch, args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    num_classes=args.num_classes)

            scheduler.step()

            if best_val_acc > 0.9:
                count = True
            # validate
            if os.path.exists("./val_Error statistics/{}".format(model_name)) is False:
                os.makedirs("./val_Error statistics/{}".format(model_name))
            excel_path = "./val_Error statistics/{}/{}.xlsx".format(model_name, epoch)
            val_loss, val_acc = evaluate(model=model,
                                         model_name=model_name,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch,
                                         excel_path=excel_path,
                                         num_classes=args.num_classes,
                                         count=count)
            log_dir = "./runs/{}/".format(model_name)
            tb_writer = SummaryWriter(log_dir)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                if os.path.exists("./weights/{}".format(model_name)) is False:
                    os.makedirs("./weights/{}".format(model_name))
                torch.save(model.state_dict(), "./weights/{}/best_train_model.pth".format(model_name))
            if val_acc > best_val_acc:
                print("正确率上升！")
                best_val_acc = val_acc
                if os.path.exists("./weights/{}".format(model_name)) is False:
                    os.makedirs("./weights/{}".format(model_name))
                torch.save(model.state_dict(), "./weights/{}/best_val_model.pth".format(model_name))

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    "best_metric": val_acc,
                    "model_name": model_name,
                }
                if os.path.exists("./best_val_checkpoint/{}".format(model_name)) is False:
                    os.makedirs("./best_val_checkpoint/{}".format(model_name))

                ckpt_name = "./best_val_checkpoint/{}/best_val_checkpoint.{}".format(model_name, CHECKPOINT_EXTN)
                torch.save(checkpoint, ckpt_name) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=str, default="m")
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.001)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default='/Users/wangzhaojiang/Desktop/六体文字识别/ml-cvnets-main/datasets/train'  )
    parser.add_argument('--datasets_count', type=bool, default=False)
    parser.add_argument('--weights', type=str, default="", help='initial weights path')
    parser.add_argument('--checkpoint', type=str, default="./best_val_checkpoint", help='continue training from checkpoint')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
