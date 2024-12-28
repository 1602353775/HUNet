
import os
import cv2
import json
import numpy as np
import onnxruntime
from PIL import Image

# 定义全局变量
HanZi = None  

def initialize_list():
    global HanZi
    # 在此处初始化列表
    json_path = 'GF_class_indices.json'

    with open(json_path, "r") as f:
        HanZi = json.load(f)

# 初始化汉字索引列表
initialize_list()
# 检查是否有可用的GPU
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    # 配置会话以使用GPU
    session_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession("results/onnx/FLENet_T2.onnx", session_options, providers=['CUDAExecutionProvider'])
    print("Using GPU for inference")
else:
    # 没有可用的GPU，使用CPU
    session = onnxruntime.InferenceSession("results/onnx/FLENet_T2.onnx")
    print("Using CPU for inference")

def test_transforms(img,padding):
    return cv2.copyMakeBorder(img,padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT, value=(255, 255, 255))

def normalize(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img

def normalize_image(image):
    # 将图像转换为浮点型数据类型
    image_float = image.astype(np.float32)
    # 将图像数组除以255，将像素值缩放到0-1之间
    normalized_image = image_float / 255.0
    # 对每个通道进行归一化
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    normalized_image = (normalized_image - mean) / std
    return normalized_image.astype(np.float32)

def  Resize (image,size:int=224):
    try:
        height = image.shape[0]
        width = image.shape[1]
        ratio = height / width
        if height > width:
            height = size
            width = int(height / ratio)
        else:
            width = size
            height = int(width * ratio)
        # 检查新的width和height是否大于0
        if width <= 0 or height <= 0:
            raise ValueError('Width and height must be greater than 0')
        image = cv2.resize(image,(width,height),interpolation= cv2.INTER_LINEAR)
        # cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR) # 按照比例因子缩放
        if height > width:
            left_size = int((height - width) / 2)
            right_size = height - width -left_size
            image = cv2.copyMakeBorder(image, 0, 0, left_size, right_size,cv2.BORDER_CONSTANT, value=(255,255,255))
        else:
            top_size = int((width - height) / 2)
            bottom_size = width -height - top_size
            image = cv2.copyMakeBorder(image, top_size, bottom_size, 0, 0,cv2.BORDER_CONSTANT, value=(255,255,255))
    except Exception as e:
        # print(str(e))
        raise e
    return image

def softmax_top_five_indices_and_values(arr):
    # 对数组进行 softmax 转换
    max_val = np.max(arr)
    arr_exp = np.exp(arr - max_val)
    softmax_arr = arr_exp / np.sum(arr_exp)

    # 获取最大的五个值的索引和值
    top_five_indices = np.argsort(softmax_arr.flatten())[-5:][::-1]
    top_five_values = softmax_arr.flatten()[top_five_indices]

    # 将值转换为百分比形式，保留两位小数
    top_five_values_percent = (top_five_values * 100).round(2)

    # 构造结果字典
    result_dict = {"indices": top_five_indices.tolist(), "probability": [f"{value:.2f}%" for value in top_five_values_percent]}
    return result_dict

def format_results(input_dict, index_label_dict):
    results = []
    
    indices = input_dict['indices']
    probabilities = input_dict['probability']
    
    for i in range(len(indices)):
        index = str(indices[i])
        probability = probabilities[i]
        
        if index in index_label_dict:
            character = index_label_dict[index]
            result = {"character": character, "probability": probability}
            results.append(result)
    
    formatted_results = {"results": results}
    return formatted_results

def find_images_in_folder(folder_path):
    """
    遍历指定文件夹及其所有子文件夹，找到所有图片文件的路径。
    忽略以“.”开头的隐藏文件。
    
    参数:
    - folder_path: 字符串，要遍历的顶级文件夹的路径。
    
    返回:
    - images: 图片文件路径的列表。
    """
    # 支持的图片文件扩展名列表
    supported_image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif')
    # 存储找到的图片路径
    images = []

    # 使用os.walk遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        # 过滤掉以"."开头的隐藏文件和文件夹
        files = [f for f in files if not f.startswith('.')]
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        # 检查每个文件是否是支持的图片文件
        for file in files:
            if file.endswith(supported_image_extensions):
                # 构造完整的文件路径
                full_path = os.path.join(root, file)
                # 将图片路径添加到列表中
                images.append(full_path)

    return images

def recognize_calligraphy(img_path,img_size,img_type="binary"):
    try:
        img = cv2.imread(img_path)
        if img_type == "binary":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            retval, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img = Resize(img,img_size)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = Resize(img,img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = normalize_image(img)
        img = np.transpose(img, (2, 0, 1))
        input_array = np.expand_dims(img, axis=0)
        input_data = {'modelInput': input_array}
        output_names = session.get_outputs()
        output_names = [output.name for output in output_names]
        outputs = session.run(output_names, input_data, None)
        top_5_result = softmax_top_five_indices_and_values(outputs[0])
        results = format_results(top_5_result,HanZi)
    
    except Exception as e:
        # print(str(e))
        raise e
    return results

if __name__ == '__main__':
    img_size = 128
    datasets = "dataset/test_1"
    test_imgs = find_images_in_folder(datasets)
    print("共有测试数据：",len(test_imgs))
    for img in test_imgs:
        predict = recognize_calligraphy(img,img_size)
        print(img,"识别为：",predict['results'][0]['character'])
    