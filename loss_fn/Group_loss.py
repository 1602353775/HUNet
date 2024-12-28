import torch
import torch.nn as nn
from loss_fn.GCL_loss import GCLLoss


def group_loss(group_labels, group_logits, labels_per_group, multi_head_logits, r=0.5):
    # labels = torch.stack([group_labels, labels_per_group], dim=1)
    criterion = nn.CrossEntropyLoss()  # 计算交叉熵损失
    group_loss = criterion(group_logits, group_labels)
    group_preds = torch.max(group_logits, dim=1)[1]
    group_acc_num = torch.eq(group_preds, group_labels).sum()

    multi_head_losses = []
    hanzi_acc_num = torch.zeros(4).to('cuda')
    for i in range(len(multi_head_logits)):
        g_logits = multi_head_logits[i]
        g_labels = labels_per_group[i]
        if len(g_labels)>0:
            g_loss = criterion(g_logits,g_labels)
            multi_head_losses.append(g_loss)
            zi_preds = torch.max(g_logits, dim=1)[1]
            zi_acc_num = torch.eq(zi_preds, g_labels).sum()
            hanzi_acc_num[i]= zi_acc_num
        else:
            hanzi_acc_num[i]= 0
    
    average_loss = torch.mean(torch.stack(multi_head_losses))
    all_loss = r * group_loss + (1 - r) * average_loss
    

    return all_loss,group_acc_num,hanzi_acc_num

def group_loss_2(multi_head_logits,labels_per_group,group_samples_lists):
    classes_per_group = [2000,2000,2000,2105]
    # labels = torch.stack([group_labels, labels_per_group], dim=1)
    criterion = nn.CrossEntropyLoss()  # 计算交叉熵损失

    multi_head_losses = []
    hanzi_acc_num = torch.zeros(1).to('cuda')
    for i in range(len(multi_head_logits)):
        g_loss = criterion(multi_head_logits[i],labels_per_group[i])
        multi_head_losses.append(g_loss)
        g_preds = torch.max(multi_head_logits[i], dim=1)[1]
        hanzi_acc = torch.zeros_like(g_preds)
        mask = torch.eq(g_preds, labels_per_group[i])
        hanzi_acc[(mask == True) & (labels_per_group[i] != classes_per_group[i])] += 1
        hanzi_acc_num +=  torch.sum(hanzi_acc)

    average_loss = torch.mean(torch.stack(multi_head_losses))
    return average_loss,hanzi_acc_num


def group_loss_3(multi_head_logits,labels_per_group,group_samples_lists):
    classes_per_group = [2000,2000,2000,2105]
    criterions = []
    for g in range(len(multi_head_logits)):
        criterion = GCLLoss(group_samples_lists[g])  # 计算交叉熵损失
        criterions.append(criterion)

    multi_head_losses = []
    hanzi_acc_num = torch.zeros(1).to('cuda')
    for i in range(len(multi_head_logits)):
        g_loss = criterions[i](multi_head_logits[i],labels_per_group[i])
        multi_head_losses.append(g_loss)
        g_preds = torch.max(multi_head_logits[i], dim=1)[1]
        hanzi_acc = torch.zeros_like(g_preds)
        mask = torch.eq(g_preds, labels_per_group[i])
        hanzi_acc[(mask == True) & (labels_per_group[i] != classes_per_group[i])] += 1
        hanzi_acc_num +=  torch.sum(hanzi_acc)

    # average_loss = torch.mean(torch.stack(multi_head_losses))
    average_loss = multi_head_losses[0]
    return average_loss,hanzi_acc_num




