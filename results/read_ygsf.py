import os


import os 
import json

def get_image_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.startswith('.') and os.path.isfile(file_path):
                ext = os.path.splitext(file)[1]
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                    image_paths.append(file_path)
    return image_paths

def ygsf (data_dir):
    json_path = './GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    class_indict = {value: key for key, value in class_indict.items()}
    
    train_images_path = []   # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    
    train_fonts = [font for font in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,font))]
    # 遍历每个文件夹下的文件
    for font in train_fonts:
        font_dir = os.path.join(data_dir,font)
        for subfolder in os.listdir(font_dir):
            if subfolder in class_indict:
                subfolder_path = os.path.join(font_dir, subfolder)
                if subfolder_path.startswith('.'):
                    continue
                if os.path.isdir(subfolder_path):
                    paths = get_image_paths(subfolder_path)
                    for path in paths:
                        train_images_path.append(path)
                        train_images_label.append(int(class_indict[subfolder]))
                    # for image_name in os.listdir(subfolder_path):
                    #     image_path = os.path.join(subfolder_path, image_name)
                    #     if image_path.startswith('.'):
                    #         continue
                    #     train_images_path.append(image_path)
                    #     # 标签转换
                    #     train_images_label.append(int(class_indict[subfolder]))
    return train_images_path,train_images_label

if __name__ == '__main__':
    data_dir = r"D:\wangzhaojiang\书法真迹\ygsf_zj\以观书法"
    train_images_path, train_images_label = ygsf (data_dir)
    print(len(train_images_label))