import time
import torch
import os 
import json
from models.Multi_FLENet import Multi_FLENet_T0

from PIL import Image
from torchvision import transforms as T
from typing import  Tuple, Union
from experiments.Ensemble_learning.utils import train_one_epoch, evaluate_one_epoch,read_data


train_path = r"C:\Users\wzj\Desktop\ks_mobilevit\datasets\train"
test_path = r"C:\Users\wzj\Desktop\ks_mobilevit\datasets\test"
train_images_path, train_image_group,train_images_label, test_images_path, test_image_group, test_images_label = read_data(train_path,test_path)



def test_transforms(padding:Union[Tuple, int]):
    return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                        T.ToTensor(),
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # read class_indict
    json_path = '../class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    img_size = 128

    acc_num = torch.zeros(1).to(device)
    for i in range(len(test_images_path)):
        img = Image.open(test_images_path[i]).convert('RGB')
        width, height = img.size
        if height>width:
            new_width = int(img_size*(width/height))
            img = img.resize((new_width,img_size),resample=Image.Resampling.BILINEAR)
            l_padding = int((img_size-new_width)/2)
            r_padding = int(img_size-l_padding-new_width)
            padding = (l_padding,0,r_padding,0)
        else:
            new_height = int(img_size*(height/width))
            img = img.resize((img_size, new_height),resample=Image.Resampling.BILINEAR)
            t_padding = int((img_size - new_height) / 2)
            b_padding = int(img_size - t_padding - new_height)
            padding = (0,t_padding,0,b_padding)

        transform_fn = test_transforms(padding=padding)
        img = transform_fn(img)
        img = torch.unsqueeze(img, dim=0)
        model = Multi_FLENet_T0(num_classes=8105).to(device)

        model_weight_path = r"./experiments/Ensemble_learning/results/weights/Multi_FLENet_T0/best_val_model.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        
        with torch.no_grad():
            # predict class
            try:
                start_time = time.time()
                preds = model(img.to(device),train=False)
                end_time = time.time()
                infer_time = end_time - start_time
                print(f"Inference time: {infer_time:.4f} seconds",preds)
            except BaseException as e:
                print("%s无法识别！"%test_images_path[i])
                print(e.args)
        # with torch.no_grad():
        #     # predict class
        #     try:
        #         start_time = time.time()
        #         preds = model(img.to(device),train=False)
        #         end_time = time.time()
        #         infer_time = end_time - start_time
        #         acc_num += torch.eq(preds, test_labels[i]).sum()
        #         preds = preds.numpy()
        #         print(f"Inference time: {infer_time:.4f} seconds",class_indict[str(test_labels[i])],class_indict[str(preds[0])])
        #     except BaseException as e:
        #         print("%s无法识别！"%test_images[i])
        #         print(e.args)
test()