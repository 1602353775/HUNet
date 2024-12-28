from PIL import Image
import torch
import numpy
import io
import cv2
from torch.utils.data import Dataset
import psutil
import random
from torchvision import transforms as T
from typing import Optional, Tuple, Dict, List, Union
import numpy as np


# 设置随机数种子
seed = 2023
torch.manual_seed(seed)
random.seed(seed)


# 训练单个分类器
class MyDataSet_1(Dataset):
    """自定义数据集"""

    def __init__(self, images_paths: list, images_labels: list, img_size=128, img_type ='Binary_image', is_training: Optional[bool] = True,dataset = "test"):
        self.paths = images_paths
        self.labels = images_labels
        self.img_size = img_size
        self.img_type = img_type
        self.is_training = is_training
        self.cached_data = ( dict() )
        self.cache_limit = 80.0
        self.dataset = dataset

    def __len__(self):
        return len(self.paths)

    def read_image_pil(self, path: str, *args, **kwargs):
        def convert_to_rgb(inp_data: Union[str, io.BytesIO]):
            try:
                rgb_img = Image.open(inp_data).convert("RGB")
            except:
                rgb_img = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))
            return rgb_img

        if self.cached_data is not None:
            # code for caching data on RAM
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                img_byte = self.cached_data[path]

            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                # image is not present in cache and RAM usage is less than the threshold, add to cache
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)
                    self.cached_data[path] = img_byte
            else:
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)  # in-memory data
            img = convert_to_rgb(img_byte)
        else:
            img = convert_to_rgb(path)
        return img

    def add_sp_noise(self,image, prob=0.1):
        data = np.asarray(image)
        shape = data.shape
        noise = np.random.random(shape)  # 生成随机值
        data[noise < prob] = 0  # 小于 probability 的随机数代表黑色斑点
        data[noise > 1 - prob] = 255  # 大于 1-probability 的随机数代表白色斑点
        return Image.fromarray(data)

    def process_image_in_chunks_with_noise(self,image_path, w, h):
        chunks = []
        with Image.open(image_path) as img:
            img = img.resize((256, 256))
            width, height = img.size
            chunk_width = width // w
            chunk_height = height // h
            for i in range(h):
                top = i * chunk_height
                bottom = (i + 1) * chunk_height if i < h - 1 else height
                for j in range(w):
                    left = j * chunk_width
                    right = (j + 1) * chunk_width if j < w - 1 else width
                    chunk = (left, top, right, bottom)
                    chunks.append(chunk)

            chunk_images = []
            for chunk in chunks:
                prob = round(random.uniform(0, 0.5), 2)
                chunk_image = self.add_sp_noise(img.crop(chunk), prob)
                chunk_images.append(chunk_image)

            # 将处理后的图像块组合成一个完整的图像
            # print(len(chunk_images))
            result = Image.new(img.mode, (width, height))
            for i, chunk_image in enumerate(chunk_images):
                left = (i % w) * chunk_width
                right = ((i % w) + 1) * chunk_width if (i % w) < w - 1 else width
                top = (i // w) * chunk_height
                bottom = ((i // w) + 1) * chunk_height if (i // w) < h - 1 else height
                result.paste(chunk_image, (left, top, right, bottom))

        # 返回完整的处理后的图像
        return result

    def binarization(self, pil_img):
        # 将 PIL 图片转为 OpenCV 格式图像
        img_cv = cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGBA2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        height, width = binary.shape
        corner_values = [binary[0, 0], binary[0, width-1], binary[height-1, 0], binary[height-1, width-1]]
        # 计算出角落处的黑色像素数量
        black_count = corner_values.count(0)

        if black_count >= 3:
            # 转换为三通道灰度图像并取反
            binary_gray = cv2.bitwise_not(binary)
            binary_rgb = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2RGB)
            # 将 OpenCV 格式的图像转为 PIL 格式图像
            return Image.fromarray(binary_rgb.astype('uint8'))
        else:
            # 转换为三通道灰度图像
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            # 将 OpenCV 格式的图像转为 PIL 格式图像
            return Image.fromarray(binary_rgb.astype('uint8'))

    def _training_transforms(self, size: Union[Tuple, int], padding:Union[Tuple, int], img_type:str):
        if img_type == 'Binary_image':
            kernel_size = 2*random.randint(2, 10)+1
            return T.Compose([T.Pad(padding, fill=255, padding_mode='edge'),
                              T.RandomApply([T.RandomRotation(degrees=20, fill=255)], p=0.1),
                              T.RandomApply([T.GaussianBlur(kernel_size, sigma=(5,40))], p=0.2),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])
        else:
            return T.Compose([T.Pad(padding, fill=255, padding_mode='edge'),
                              T.RandomApply([T.RandomRotation(degrees=20, fill=255)], p=0.1),
                              T.RandomApply([T.ColorJitter(brightness=(2,4), contrast=(1,3), saturation=(2,4), hue=0)], p=0.2),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

    def _validation_transforms(self, size: Union[Tuple, int], padding:Union[Tuple, int], img_type:str):
        if img_type == 'Binary_image':
            return T.Compose([T.Pad(padding, fill=255, padding_mode='edge'),  # constant edge reflect symmetric
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

        else:
            return T.Compose([T.Pad(padding, fill=255, padding_mode='edge'),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

    def __getitem__(self, item):
        img = self.read_image_pil(self.paths[item])

        width, height = img.size
        if height>width:
            new_width = int(self.img_size*(width/height))
            img = img.resize((new_width,self.img_size),resample=Image.Resampling.BILINEAR)
            l_padding = int((self.img_size-new_width)/2)
            r_padding = int(self.img_size-l_padding-new_width)
            padding = (l_padding,0,r_padding,0)
        else:
            new_height = int(self.img_size*(height/width))
            img = img.resize((self.img_size, new_height),resample=Image.Resampling.BILINEAR)
            t_padding = int((self.img_size - new_height) / 2)
            b_padding = int(self.img_size - t_padding - new_height)
            padding = (0,t_padding,0,b_padding)

        if self.is_training and self.img_type =='Binary_image':
            transform_fn = self._training_transforms(size=self.img_size, padding=padding, img_type=self.img_type)
        elif self.is_training and self.img_type =='RGB_image':
            img = self.binarization(img)
            transform_fn = self._training_transforms(size=self.img_size, padding=padding, img_type=self.img_type)
        else:
            # same for validation and evaluation
            if self.dataset == "mact":
                img = self.binarization(img)
                transform_fn = self._validation_transforms(size=self.img_size, padding=padding, img_type=self.img_type)
            else:
                transform_fn = self._validation_transforms(size=self.img_size, padding=padding, img_type=self.img_type)

        label = self.labels[item]
        path = self.paths[item]
        img = transform_fn(img)

        return img, label, path

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, paths = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, paths  # 返回元组


# 训练多个分类器
class MyDataSet_2(Dataset):
    """自定义数据集"""

    def __init__(self, images_paths: list, images_labels: list, images_levels:list, img_size=256, img_type ='Binary_image', is_training: Optional[bool] = True):
        self.paths = images_paths
        self.labels = images_labels
        self.images_levels = images_levels 
        self.img_size = img_size
        self.img_type = img_type
        self.is_training = is_training
        self.cached_data = ( dict() )
        self.cache_limit = 80.0

    def __len__(self):
        return len(self.paths)

    def read_image_pil(self, path: str, *args, **kwargs):
        def convert_to_rgb(inp_data: Union[str, io.BytesIO]):
            try:
                rgb_img = Image.open(inp_data).convert("RGB")
            except:
                rgb_img = None
            return rgb_img

        if self.cached_data is not None:
            # code for caching data on RAM
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                img_byte = self.cached_data[path]

            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                # image is not present in cache and RAM usage is less than the threshold, add to cache
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)
                    self.cached_data[path] = img_byte
            else:
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)  # in-memory data
            img = convert_to_rgb(img_byte)
        else:
            img = convert_to_rgb(path)
        return img
    
    def add_sp_noise(self,image, prob=0.2):
        data = np.asarray(image)
        shape = data.shape
        noise = np.random.random(shape)  # 生成随机值
        data[noise < prob] = 0  # 小于 probability 的随机数代表黑色斑点
        data[noise > 1 - prob] = 255  # 大于 1-probability 的随机数代表白色斑点
        return Image.fromarray(data)

    def process_image_in_chunks_with_noise(self,image_path, w, h):
        chunks = []
        with Image.open(image_path) as img:
            img = img.resize((256, 256))
            width, height = img.size
            chunk_width = width // w
            chunk_height = height // h
            for i in range(h):
                top = i * chunk_height
                bottom = (i + 1) * chunk_height if i < h - 1 else height
                for j in range(w):
                    left = j * chunk_width
                    right = (j + 1) * chunk_width if j < w - 1 else width
                    chunk = (left, top, right, bottom)
                    chunks.append(chunk)

            chunk_images = []
            for chunk in chunks:
                prob = round(random.uniform(0, 0.5), 2)
                chunk_image = self.add_sp_noise(img.crop(chunk), prob)
                chunk_images.append(chunk_image)

            # 将处理后的图像块组合成一个完整的图像
            print(len(chunk_images))
            result = Image.new(img.mode, (width, height))
            for i, chunk_image in enumerate(chunk_images):
                left = (i % w) * chunk_width
                right = ((i % w) + 1) * chunk_width if (i % w) < w - 1 else width
                top = (i // w) * chunk_height
                bottom = ((i // w) + 1) * chunk_height if (i // w) < h - 1 else height
                result.paste(chunk_image, (left, top, right, bottom))

        # 返回完整的处理后的图像
        return result

    def binarization(self, pil_img):
        # 将 PIL 图片转为 OpenCV 格式图像
        img_cv = cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGBA2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        height, width = binary.shape
        corner_values = [binary[0, 0], binary[0, width-1], binary[height-1, 0], binary[height-1, width-1]]
        # 计算出角落处的黑色像素数量
        black_count = corner_values.count(0)

        if black_count >= 3:
            # 转换为三通道灰度图像并取反
            binary_gray = cv2.bitwise_not(binary)
            binary_rgb = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2RGB)
            # 将 OpenCV 格式的图像转为 PIL 格式图像
            return Image.fromarray(binary_rgb.astype('uint8'))
        else:
            # 转换为三通道灰度图像
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            # 将 OpenCV 格式的图像转为 PIL 格式图像
            return Image.fromarray(binary_rgb.astype('uint8'))

    def _training_transforms(self, size: Union[Tuple, int], padding:Union[Tuple, int], img_type:str):
        if img_type == 'Binary_image':
            kernel_size = 2*random.randint(2, 10)+1
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.RandomApply([T.RandomRotation(degrees=20, fill=255)], p=0.1),
                              T.RandomApply([T.GaussianBlur(kernel_size, sigma=(5,40))], p=0.2),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])
        else:
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.RandomApply([T.RandomRotation(degrees=20, fill=255)], p=0.1),
                              T.RandomApply([T.ColorJitter(brightness=(2,4), contrast=(1,3), saturation=(2,4), hue=0)], p=0.2),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

    def _validation_transforms(self, size: Union[Tuple, int], padding:Union[Tuple, int], img_type:str):
        if img_type == 'Binary_image':
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

        else:
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

    def __getitem__(self, item):
        img = self.read_image_pil(self.paths[item])
    
        width, height = img.size
        if height>width:
            new_width = int(self.img_size*(width/height))
            img = img.resize((new_width,self.img_size),resample=Image.Resampling.BILINEAR)
            l_padding = int((self.img_size-new_width)/2)
            r_padding = int(self.img_size-l_padding-new_width)
            padding = (l_padding,0,r_padding,0)
        else:
            new_height = int(self.img_size*(height/width))
            img = img.resize((self.img_size, new_height),resample=Image.Resampling.BILINEAR)
            t_padding = int((self.img_size - new_height) / 2)
            b_padding = int(self.img_size - t_padding - new_height)
            padding = (0,t_padding,0,b_padding)

        if self.is_training:
            transform_fn = self._training_transforms(size=self.img_size, padding=padding, img_type=self.img_type)
        else:
            # same for validation and evaluation
            transform_fn = self._validation_transforms(size=self.img_size, padding=padding, img_type=self.img_type)

        label = self.labels[item]
        level = self.images_levels[item]
        img = transform_fn(img)
        path = self.paths[item]

        return img, label, level, path

# 分组训练
class MyDataSet_3(Dataset):
    """自定义数据集"""

    def __init__(self, images_paths: list, images_labels: list, images_levels:list, img_size=256, img_type ='Binary_image', is_training: Optional[bool] = True):
        self.paths = images_paths
        self.labels = images_labels
        self.images_levels = images_levels 
        self.img_size = img_size
        self.img_type = img_type
        self.is_training = is_training
        self.cached_data = ( dict() )
        self.cache_limit = 80.0

    def __len__(self):
        return len(self.paths)

    def read_image_pil(self, path: str, *args, **kwargs):
        def convert_to_rgb(inp_data: Union[str, io.BytesIO]):
            try:
                rgb_img = Image.open(inp_data).convert("RGB")
            except:
                rgb_img = None
            return rgb_img

        if self.cached_data is not None:
            # code for caching data on RAM
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                img_byte = self.cached_data[path]

            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                # image is not present in cache and RAM usage is less than the threshold, add to cache
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)
                    self.cached_data[path] = img_byte
            else:
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)  # in-memory data
            img = convert_to_rgb(img_byte)
        else:
            img = convert_to_rgb(path)
        return img
    
    def add_sp_noise(self,image, prob=0.2):
        data = np.asarray(image)
        shape = data.shape
        noise = np.random.random(shape)  # 生成随机值
        data[noise < prob] = 0  # 小于 probability 的随机数代表黑色斑点
        data[noise > 1 - prob] = 255  # 大于 1-probability 的随机数代表白色斑点
        return Image.fromarray(data)

    def process_image_in_chunks_with_noise(self,image_path, w, h):
        chunks = []
        with Image.open(image_path) as img:
            img = img.resize((256, 256))
            width, height = img.size
            chunk_width = width // w
            chunk_height = height // h
            for i in range(h):
                top = i * chunk_height
                bottom = (i + 1) * chunk_height if i < h - 1 else height
                for j in range(w):
                    left = j * chunk_width
                    right = (j + 1) * chunk_width if j < w - 1 else width
                    chunk = (left, top, right, bottom)
                    chunks.append(chunk)

            chunk_images = []
            for chunk in chunks:
                prob = round(random.uniform(0, 0.5), 2)
                chunk_image = self.add_sp_noise(img.crop(chunk), prob)
                chunk_images.append(chunk_image)

            # 将处理后的图像块组合成一个完整的图像
            print(len(chunk_images))
            result = Image.new(img.mode, (width, height))
            for i, chunk_image in enumerate(chunk_images):
                left = (i % w) * chunk_width
                right = ((i % w) + 1) * chunk_width if (i % w) < w - 1 else width
                top = (i // w) * chunk_height
                bottom = ((i // w) + 1) * chunk_height if (i // w) < h - 1 else height
                result.paste(chunk_image, (left, top, right, bottom))

        # 返回完整的处理后的图像
        return result

    def binarization(self, pil_img):
        # 将 PIL 图片转为 OpenCV 格式图像
        img_cv = cv2.cvtColor(numpy.array(pil_img), cv2.COLOR_RGBA2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        height, width = binary.shape
        corner_values = [binary[0, 0], binary[0, width-1], binary[height-1, 0], binary[height-1, width-1]]
        # 计算出角落处的黑色像素数量
        black_count = corner_values.count(0)

        if black_count >= 3:
            # 转换为三通道灰度图像并取反
            binary_gray = cv2.bitwise_not(binary)
            binary_rgb = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2RGB)
            # 将 OpenCV 格式的图像转为 PIL 格式图像
            return Image.fromarray(binary_rgb.astype('uint8'))
        else:
            # 转换为三通道灰度图像
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            # 将 OpenCV 格式的图像转为 PIL 格式图像
            return Image.fromarray(binary_rgb.astype('uint8'))

    def _training_transforms(self, size: Union[Tuple, int], padding:Union[Tuple, int], img_type:str):
        if img_type == 'Binary_image':
            kernel_size = 2*random.randint(2, 10)+1
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.RandomApply([T.RandomRotation(degrees=20, fill=255)], p=0.1),
                              T.RandomApply([T.GaussianBlur(kernel_size, sigma=(5,40))], p=0.2),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])
        else:
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.RandomApply([T.RandomRotation(degrees=20, fill=255)], p=0.1),
                              T.RandomApply([T.ColorJitter(brightness=(2,4), contrast=(1,3), saturation=(2,4), hue=0)], p=0.2),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

    def _validation_transforms(self, size: Union[Tuple, int], padding:Union[Tuple, int], img_type:str):
        if img_type == 'Binary_image':
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

        else:
            return T.Compose([T.Pad(padding, fill=255, padding_mode='symmetric'),
                              T.ToTensor(),
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

    def __getitem__(self, item):
        img = self.read_image_pil(self.paths[item])
    
        width, height = img.size
        if height>width:
            new_width = int(self.img_size*(width/height))
            img = img.resize((new_width,self.img_size),resample=Image.Resampling.BILINEAR)
            l_padding = int((self.img_size-new_width)/2)
            r_padding = int(self.img_size-l_padding-new_width)
            padding = (l_padding,0,r_padding,0)
        else:
            new_height = int(self.img_size*(height/width))
            img = img.resize((self.img_size, new_height),resample=Image.Resampling.BILINEAR)
            t_padding = int((self.img_size - new_height) / 2)
            b_padding = int(self.img_size - t_padding - new_height)
            padding = (0,t_padding,0,b_padding)

        if self.is_training:
            transform_fn = self._training_transforms(size=self.img_size, padding=padding, img_type=self.img_type)
        else:
            # same for validation and evaluation
            transform_fn = self._validation_transforms(size=self.img_size, padding=padding, img_type=self.img_type)

        label = self.labels[item]
        level = self.images_levels[item]
        img = transform_fn(img)
        path = self.paths[item]

        return img, label, level, path

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, levels,paths = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        levels = torch.as_tensor(levels)
        return images, labels, levels,paths  # 返回元组