import numpy as np
from collections import Counter

def balance_sampler(image_paths, labels):
    # 计算每个类别的样本数量
    label_counter = Counter(labels)
    num_samples = len(labels)
    num_classes = len(label_counter)

    # 计算每个类别需要的样本数量
    target_num_samples = np.ceil(num_samples / num_classes)

    new_image_paths = []
    new_labels = []

    for c in label_counter:
        # 找到当前类别的所有样本的索引
        indices = np.where(np.array(labels) == c)[0]
        num_current_samples = len(indices)

        # 如果当前类别的样本数量小于目标数量，进行过采样
        if num_current_samples < target_num_samples:
            # 计算需要从原始样本中采样的数量
            num_new_samples = int(target_num_samples - num_current_samples)
            # 随机从原始样本中采样
            sample_indices = np.random.choice(indices, num_new_samples, replace=True)
            # 将采样的样本加入新的数组中
            new_image_paths += list(np.array(image_paths)[sample_indices])
            new_labels += [c] * num_new_samples
        else:
            # 如果当前类别的样本数量大于等于目标数量，不进行采样
            new_image_paths += list(np.array(image_paths)[indices])
            new_labels += [c] * num_current_samples

    return new_image_paths, new_labels
