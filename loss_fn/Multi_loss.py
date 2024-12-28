import torch
import torch.nn as nn

def multi_loss_2(level_labels, level_logits, labels, multi_logits, classes_per_level, r=0.9):
    criterion = nn.CrossEntropyLoss()  # 计算交叉熵损失
    level_loss = criterion(level_logits, level_labels)
    
    level_preds = torch.max(level_logits, dim=1)[1]
    level_acc_num = torch.eq(level_preds, level_labels).sum()
    level_1_pre_index = torch.eq(level_preds, level_labels) & (level_labels == 0)
    level_2_pre_index = torch.eq(level_preds, level_labels) & (level_labels == 1)
    level_3_pre_index = torch.eq(level_preds, level_labels) & (level_labels == 2)
    # 特判：若某个 level 的预测标签全部错误，则对应的 labels 和 logits 可能会为空张量
    # 在这种情况下直接返回 level_loss 即可，不用计算 average_loss
    if not level_1_pre_index.any() and not level_2_pre_index.any() and not level_3_pre_index.any():
        return level_loss,0,0
    
    level_1_labels = labels[level_1_pre_index]
    level_2_labels = labels[level_2_pre_index] - classes_per_level[0]
    level_3_labels = labels[level_3_pre_index] - (classes_per_level[0] + classes_per_level[1])
    
    level_1_logits = multi_logits[0][level_1_pre_index]
    level_2_logits = multi_logits[1][level_2_pre_index]
    level_3_logits = multi_logits[2][level_3_pre_index]

    # 如果某个 level 的 labels 或 logits 为空，则直接忽略该 level 的损失
    # 因为在这种情况下无法计算该 level 的损失
    level_losses = []
    hanzi_acc_num = 0
    if level_1_labels.nelement() > 0 and level_1_logits.nelement() > 0:
        level_1_loss = criterion(level_1_logits, level_1_labels)
        level_losses.append(level_1_loss)
        pred_hanzis = torch.max(level_1_logits, dim=1)[1]
        hanzi_acc_num += torch.eq(pred_hanzis, level_1_labels).sum()
    if level_2_labels.nelement() > 0 and level_2_logits.nelement() > 0:
        level_2_loss = criterion(level_2_logits, level_2_labels)
        level_losses.append(level_2_loss)
        pred_hanzis = torch.max(level_2_logits, dim=1)[1]
        hanzi_acc_num += torch.eq(pred_hanzis, level_2_labels).sum()
    if level_3_labels.nelement() > 0 and level_3_logits.nelement() > 0:
        level_3_loss = criterion(level_3_logits, level_3_labels)
        level_losses.append(level_3_loss)
        pred_hanzis = torch.max(level_3_logits, dim=1)[1]
        hanzi_acc_num += torch.eq(pred_hanzis, level_3_labels).sum()

    if not level_losses:  # 所有的 level_labels 和 multi_logits 中均无有效数据
        return torch.tensor(float('nan'))  # 返回 nan 损失即可

    average_loss = torch.mean(torch.stack(level_losses))
    
    return r * level_loss + (1 - r) * average_loss,level_acc_num,hanzi_acc_num

def multi_loss(level_labels, level_logits, labels, multi_head_logits, classes_per_level, r=0.5):
    criterion = nn.CrossEntropyLoss()  # 计算交叉熵损失
    level_loss = criterion(level_logits, level_labels)
    level_preds = torch.max(level_logits, dim=1)[1]
    level_acc_num = torch.eq(level_preds, level_labels).sum()
    
    multi_head_labels = []
    offset = 0
    for i in range(len(classes_per_level)):
        if i==0:
            new_labels = torch.where(labels >= classes_per_level[i], torch.tensor(classes_per_level[i]), labels)
            multi_head_labels.append(new_labels)
        else:
            offset+=classes_per_level[i-1]
            new_labels = torch.where((labels >= offset) & (labels < classes_per_level[i]+offset), labels - offset, torch.tensor(classes_per_level[i]))
            multi_head_labels.append(new_labels)
       
    multi_head_losses = []
    hanzi_acc_num = 0
    for i in range(len(classes_per_level)):
        head_loss = criterion(multi_head_logits[i],multi_head_labels[i])
        multi_head_losses.append(head_loss)
        pred_hanzis = torch.max(multi_head_logits[i], dim=1)[1]
        hanzi_acc_num += torch.eq(pred_hanzis, multi_head_labels[i]).sum()

    average_loss = torch.mean(torch.stack(multi_head_losses))
    
    return r * level_loss + (1 - r) * average_loss,level_acc_num,hanzi_acc_num/len(classes_per_level)

if __name__ == '__main__':
    for i in range(10):
        level_num = 3
        no_of_classes = 20
        batch_size = 20
        classes_per_level = [10,10,10]
        input = torch.randn(batch_size, 3, 128, 128)
        level_logits = torch.rand(batch_size,level_num).float()

        multi_logits = []
        labels = torch.randint(low=0, high=no_of_classes, size=(batch_size,)) 
        level_labels = torch.zeros_like(labels) + 2   # 初始化为2
        level_labels[labels < 6] = 0
        level_labels[(labels >= 6) & (labels < 12)] = 1 
        leve_1_logits = torch.rand(batch_size,classes_per_level[0]).float()
        multi_logits.append(leve_1_logits)
        leve_2_logits = torch.rand(batch_size,classes_per_level[1]).float()
        multi_logits.append(leve_2_logits)
        leve_3_logits = torch.rand(batch_size,classes_per_level[2]).float()
        multi_logits.append(leve_3_logits)
        
        loss,level_acc_num,hanzi_acc_num = multi_loss(level_labels,level_logits,labels,multi_logits,classes_per_level,r=0.9)
        print(loss,level_acc_num,hanzi_acc_num)






