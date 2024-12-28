
import os
import json
import time
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms as T
from typing import  Tuple, Union
from shiftvit import ShiftViT_T0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ShiftViT_T0().to(device)
model_weight_path = r"D:\wangzhaojiang\FLENet\experiments\Model_architecture\results\weights\ShiftViT_T0\best_val_model.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


json_path = r"D:\wangzhaojiang\FLENet\experiments\Model_architecture\GF_class_indices.json"
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as f:
    class_indict = json.load(f)

def test_transforms(padding:Union[Tuple, int]):
    return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                        T.ToTensor(),
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
def predict(inp):

    img_size = 128
    width, height = inp.size
    if height>width:
        new_width = int(img_size*(width/height))
        inp = inp.resize((new_width,img_size),resample=Image.Resampling.BILINEAR)
        l_padding = int((img_size-new_width)/2)
        r_padding = int(img_size-l_padding-new_width)
        padding = (l_padding,0,r_padding,0)
    else:
        new_height = int(img_size*(height/width))
        inp = inp.resize((img_size, new_height),resample=Image.Resampling.BILINEAR)
        t_padding = int((img_size - new_height) / 2)
        b_padding = int(img_size - t_padding - new_height)
        padding = (0,t_padding,0,b_padding)

    transform_fn = test_transforms(padding=padding)
    img = transform_fn(inp)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device)))
        prediction = torch.softmax(output, dim=0)
        confidences = {class_indict[str(i)]: float(prediction[i]) for i in range(len(class_indict))}    
    
    return confidences

def main():
    gr.Interface(fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=5),
        examples=[r"D:\wangzhaojiang\FLENet\deploy\examples\3_停云馆法帖.png",
                  r"D:\wangzhaojiang\FLENet\deploy/examples\4_辛巳十月夜书.png",
                  r"D:\wangzhaojiang\FLENet\deploy/examples\11_急就章.png",
                  r"D:\wangzhaojiang\FLENet\deploy/examples\19_郁冈斋墨妙法帖.png",
                  r"D:\wangzhaojiang\FLENet\deploy/examples\28_刘园集帖.png"]).launch(share=True)

if __name__ == '__main__':
    main()