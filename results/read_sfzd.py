import os 
import json



def sfzd (data_dir):
    json_path = './GF_class_indices.json'

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    class_indict = {value: key for key, value in class_indict.items()}
    
    train_images_path=[]
    train_images_label=[]
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if subfolder_path.startswith('.'):
            continue
        if subfolder in class_indict:
            for font in os.listdir(subfolder_path):
                fontfolder_path = os.path.join(subfolder_path, font)
                if fontfolder_path.startswith('.') or font=="矢量大图":
                    continue
                if os.path.isdir(fontfolder_path):
                    for image_name in os.listdir(fontfolder_path):
                        image_path = os.path.join(fontfolder_path, image_name)
                        if image_path.startswith('.'):
                            continue
                        train_images_path.append(image_path)
                        # 标签转换
                        train_images_label.append(int(class_indict[subfolder]))

    return train_images_path, train_images_label

if __name__ == '__main__':
    data_dir = r"D:\wangzhaojiang\书法真迹\sfzd_zj"
    train_images_path, train_images_label = sfzd (data_dir)
    print(len(train_images_label))