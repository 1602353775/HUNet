import torch
import onnxruntime
import onnx
from PIL import Image
from torchvision import transforms as T
from models.FLENet import FLENet_T0
from models.CNNs.mobilenet_v3 import mobilenet_v3_large
from models.CNNs.RepViT import repvit_m1

from typing import  Tuple, Union
import numpy as np
import time



def pth_to_onnx ():
    model = FLENet_T0(num_classes=8105)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_weight_path = "experiments/Model_architecture/results/weights/mobilenet_v3_large/best_val_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device),strict=False)
    model.eval()
    batch_size=10
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    y = model(x)

    # Export the model
    export_onnx_file = "results/onnx/mobilenet_v3_large.onnx"
    torch.onnx.export(model,                   # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    export_onnx_file,          # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
 
    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores, boxes

def test_transforms(padding:Union[Tuple, int]):
    return T.Compose([T.Pad(padding, fill=255, padding_mode='edge'),
                        T.ToTensor(),
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])

def main():
    pth_to_onnx ()
    inp_data = "5.jpg"
    img = Image.open(inp_data).convert("RGB")
    img_size = 224
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

    # 随机生成一个大小为（50，3，128，128）的张量
    input_tensor = torch.randn(10, 3, 224, 224)

    model = FLENet_T0(num_classes=8105)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_weight_path = "experiments/Model_architecture/results/weights/mobilenet_v3_large/best_val_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device),strict=False)
    model.eval()
    start_time = time.time()
    pred = model(img)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    # print(pred[1].shape)
    pred_classes = torch.max(pred, dim=1)[1].numpy()
    print(pred_classes)
    # 将张量转换为NumPy数组
    input_array = input_tensor.numpy()
    # 创建包含输入数据的字典
    input_data = {'input': input_array}

    session = onnxruntime.InferenceSession("mobilenet_v3_large.onnx")
    output_names = session.get_outputs()
    output_names = [output.name for output in output_names]
    start_time = time.time()
    outputs = session.run(output_names, input_data, None)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    # outputs = session.run(None, [input_data])
    print(np.argmax(outputs[0], axis=1))
    print(type(np.argmax(outputs[0], axis=1)))
    # onnx_model = ONNXModel("/Volumes/wangzhaojiang/FLENet/FLENet_T0.onnx")
    # out = onnx_model.forward(img)

def test():
    # 加载 PyTorch 模型
    # model = mobilenet_v3_large(num_classes=8105)
    model = repvit_m1(num_classes=8105)
    checkpoint = torch.load(r'experiments/Model_architecture/results/best_val_checkpoint/RepViT_M1/best_val_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 定义输入和输出张量的名称和形状
    input_names = ["input"]
    output_names = ["output"]
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    
    # 将 PyTorch 模型转换为 ONNX 格式
    torch.onnx.export(
        model,  # 要转换的 PyTorch 模型
        torch.randn(input_shape),  # 模型输入的随机张量
        "repvit_m1_v10.onnx",  # 保存的 ONNX 模型的文件名
        opset_version=10,
        input_names=input_names,  # 输入张量的名称
        output_names=output_names,  # 输出张量的名称
        dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}}  # 动态轴，即输入和输出张量可以具有不同的批次大小
    )
    
    # 加载 ONNX 模型
    # onnx_model = onnx.load("mobilenet_v3_large.onnx")
    # onnx_model_graph = onnx_model.graph
    # onnx_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    onnx_session = onnxruntime.InferenceSession("repvit_m1.onnx")
    
    # 使用随机张量测试 ONNX 模型
    x = torch.randn(input_shape).numpy()
    onnx_output = onnx_session.run(output_names, {input_names[0]: x})[0]
    
    print(f"PyTorch output: {model(torch.from_numpy(x)).detach().numpy()[0, :20]}")
    print(f"ONNX output: {len(onnx_output[0, :20])}")

if __name__ == "__main__":
    test()
    # main()

# CUDA is available.
# PyTorch version: 1.10.0+cu113
# CUDA version: 11.3