from PIL import Image
import torch
import numpy as np
import io
import cv2
from torch.utils.data import Dataset
import psutil
import random
from functools import lru_cache
from torchvision import transforms as T
from typing import Optional, Tuple, Dict, List, Union
from functools import lru_cache
from math import ceil

# 设置全局随机种子
def set_seed(seed=2023):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
set_seed()

class MyDataSet(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        image_labels: List[int],
        img_size: int = 128,
        image_type: str = 'binary',
        is_training: bool = True,
        dataset_name: str = "default",
        cache_limit: float = 80.0,
        chunk_size: Tuple[int, int] = (4, 4),
        noise_intensity: float = 0.3
    ):
        """
        参数：
        image_paths: 图像路径列表
        image_labels: 对应标签列表
        img_size: 目标图像尺寸
        image_type: 图像类型 ['binary', 'rgb']
        is_training: 是否训练模式
        dataset_name: 数据集名称
        cache_limit: 内存缓存阈值百分比
        chunk_size: 分块处理尺寸 (宽度分块数, 高度分块数)
        noise_intensity: 最大噪声强度（0~1）
        """
        self.paths = image_paths
        self.labels = image_labels
        self.target_size = img_size
        self.final_size = (img_size, img_size)
        self.image_type = image_type.lower()
        self.is_training = is_training
        self.dataset_name = dataset_name
        self.cache_limit = cache_limit
        self.chunk_size = chunk_size
        self.noise_intensity = noise_intensity
        
        # 初始化缓存系统
        self._cache = {}
        self._current_mem_usage = 0.0

        # 验证参数有效性
        self._validate_parameters()

    def _validate_parameters(self):
        """验证输入参数有效性"""
        assert len(self.paths) == len(self.labels), "Paths and labels length mismatch"
        assert self.image_type in ['binary', 'rgb'], "Invalid image type"
        assert 0 < self.cache_limit <= 100, "Invalid cache limit"

    def __len__(self) -> int:
        return len(self.paths)

    @lru_cache(maxsize=20000)
    def _read_image(self, path: str) -> Image.Image:
        """带缓存的图像读取方法"""
        try:
            with open(path, 'rb') as f:
                img_data = io.BytesIO(f.read())
            return Image.open(img_data).convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            return Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))

    def _manage_cache(self, path: str):
        """智能缓存管理系统"""
        current_mem = psutil.virtual_memory().percent
        
        if path not in self._cache and current_mem <= self.cache_limit:
            img = self._read_image(path)
            self._cache[path] = img
        elif path in self._cache:
            return self._cache[path]
        
        return self._read_image(path)

    def add_binary_sp_noise(self, image: Image.Image, prob: float = 0.1) -> Image.Image:
        """优化后的二值图像椒盐噪声添加方法
        
        参数：
            image: PIL图像对象（单通道或RGB格式）
            prob: 单边噪声概率（总概率为2*prob）
        
        返回：
            添加噪声后的PIL图像对象
        """
        # 转换为单通道灰度图
        gray_img = image.convert('L')
        np_img = np.array(gray_img)
        
        # 生成随机掩码（向量化操作）
        rand_mask = np.random.random(np_img.shape)
        
        # 同时处理黑白噪声（避免中间拷贝）
        noisy_img = np.where(rand_mask < prob, 0,
                np.where(rand_mask > 1 - prob, 255, np_img))
        
        # 保持二值图像特性
        return Image.fromarray(noisy_img.astype(np.uint8)).convert('RGB')

    def _apply_chunked_noise_augmentation(self, img: Image.Image) -> Image.Image:
        """优化的分块噪声增强方法
        
        参数：
            img: 原始PIL图像对象
        
        返回：
            处理后的PIL图像对象
        """
        # 预计算分块参数
        original_size = img.size
        w, h = self.chunk_size
        width, height = img.size
        chunk_w = ceil(width / w)
        chunk_h = ceil(height / h)
        
        # 预生成所有分块坐标
        chunks = [
            (left, top, min(left+chunk_w, width), min(top+chunk_h, height))
            for top in range(0, height, chunk_h)
            for left in range(0, width, chunk_w)
        ]
        
        # 批量处理分块
        processed_chunks = []
        for chunk in chunks:
            noise = np.random.random()
            if random.random() < noise:
                noise_prob = random.uniform(0, self.noise_intensity)
                cropped = img.crop(chunk)
                processed_chunks.append(self.add_binary_sp_noise(cropped, noise_prob))
            else:
                processed_chunks.append(img.crop(chunk))
        
        # 高效拼接图像
        result = Image.new('RGB', (width, height))
        for i, chunk_img in enumerate(processed_chunks):
            col = i % w
            row = i // w
            x = col * chunk_w
            y = row * chunk_h
            result.paste(chunk_img, (x, y))
        
        return result.resize(original_size, Image.BILINEAR)

    def _adaptive_resize(self, img: Image.Image) -> Image.Image:
        """保持长宽比的智能缩放"""
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.BILINEAR)

    def _smart_padding(self, img: Image.Image) -> Tuple[Image.Image, Tuple[int]]:
        """智能填充策略"""
        w, h = img.size
        dw = self.target_size - w
        dh = self.target_size - h
        
        # 确保非负填充
        pad_left = max(dw // 2, 0)
        pad_right = max(dw - pad_left, 0)
        pad_top = max(dh // 2, 0)
        pad_bottom = max(dh - pad_top, 0)
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        padded_img = T.Pad(padding, fill=255, padding_mode='edge')(img)
        
        # 强制最终尺寸
        return padded_img.resize(self.final_size, Image.BILINEAR), padding

    def _generate_transforms(self, padding: Tuple[int]) -> T.Compose:
        """动态生成数据增强流水线"""
        transforms_list = []

        if self.is_training:
            if self.image_type == 'binary':
                transforms_list += [
                    T.RandomApply([T.RandomRotation(20, fill=255)], p=0.1),
                    T.RandomApply([T.GaussianBlur(11, sigma=(5,40))], p=0.2)
                ]
            else:
                transforms_list += [
                    T.RandomApply([T.ColorJitter(
                        brightness=(2,4), 
                        contrast=(1,3),
                        saturation=(2,4)
                    )], p=0.2)
                ]

        transforms_list += [
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ]
        
        return T.Compose(transforms_list)

    def _binarize_image(self, img: Image.Image) -> Image.Image:
        """优化后的二值化处理"""
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """核心数据获取方法"""
        path = self.paths[idx]
        
        
        # 读取并预处理图像
        # img = self._manage_cache(path)
        img = self._read_image(path)
        img = self._adaptive_resize(img)
        img, padding = self._smart_padding(img)
        
        
        # 应用图像类型特定处理
        if self.image_type == 'binary' or self.dataset_name == "mact":
            img = self._binarize_image(img)
        
        # 训练时，以0.1的概率应用分块噪声增强
        # if self.is_training and self.image_type == 'binary' and random.random() < 0.1:
        #     img = self._apply_chunked_noise_augmentation(img)
        
        # 生成动态数据增强
        transform = self._generate_transforms(padding)
        tensor_img = transform(img)
        
        assert tensor_img.shape[-2:] == torch.Size(self.final_size), \
            f"尺寸错误: {tensor_img.shape} vs {self.final_size}"
        
        
        return tensor_img, self.labels[idx], path

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """优化的批处理函数"""

        images, labels, paths = zip(*batch)
        return torch.stack(images, 0), torch.LongTensor(labels), paths
    




