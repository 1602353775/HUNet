import os
import cv2
import json
import numpy as np
import onnxruntime
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalligraphyClassifier:
    def __init__(self, model_path: str, label_path: str):
        """
        初始化分类器
        :param model_path: ONNX模型文件路径
        :param label_path: 标签JSON文件路径
        """
        self.labels = self._load_labels(label_path)
        self.session = self._init_onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def _load_labels(self, label_path: str) -> Dict[str, str]:
        """加载类别标签"""
        try:
            with open(label_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"加载标签文件失败: {str(e)}")
            raise

    def _init_onnx_session(self, model_path: str) -> onnxruntime.InferenceSession:
        """初始化ONNX推理会话"""
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # 设置推理线程数
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            session = onnxruntime.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=providers
            )
            logger.info(f"推理后端: {session.get_providers()}")
            return session
        except Exception as e:
            logger.error(f"ONNX会话初始化失败: {str(e)}")
            raise

    def _resize_and_pad(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """智能缩放并保持宽高比填充"""
        h, w = image.shape[:2]
        ratio = w / h
        
        # 计算新尺寸
        if w > h:
            new_w = target_size
            new_h = int(target_size / ratio)
        else:
            new_h = target_size
            new_w = int(target_size * ratio)
        
        # 安全检查
        new_w, new_h = max(new_w, 1), max(new_h, 1)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 计算填充
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        padding = [(pad_h, target_size - new_h - pad_h),
                   (pad_w, target_size - new_w - pad_w)]
        
        # 灰度图处理
        if len(resized.shape) == 2:
            return cv2.copyMakeBorder(resized, *padding[0], *padding[1], 
                                    cv2.BORDER_CONSTANT, value=255)
        # 彩色图处理
        return cv2.copyMakeBorder(resized, *padding[0], *padding[1], 
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """标准化图像到[-1, 1]范围"""
        image = image.astype(np.float32) / 255.0
        return (image - 0.5) / 0.5

    def _preprocess(self, img_path: str, img_size: int, is_binary: bool) -> np.ndarray:
        """图像预处理流水线（支持中文路径）"""
        # 使用二进制读取方式解决中文路径问题
        try:
            with open(img_path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"无法读取图像: {img_path} ({str(e)})")

        if image is None:
            raise ValueError(f"图像解码失败: {img_path}")

        # 二值化处理
        if is_binary:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = self._resize_and_pad(binary, img_size)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed = self._resize_and_pad(image, img_size)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # 标准化
        normalized = self._normalize_image(processed)
        return np.transpose(normalized, (2, 0, 1))  # HWC -> CHW

    def predict(self, img_path: str, img_size: int = 128, is_binary: bool = True) -> Dict[str, Any]:
        """执行书法识别"""
        try:
            # 预处理
            input_tensor = self._preprocess(img_path, img_size, is_binary)
            input_data = {self.input_name: np.expand_dims(input_tensor, axis=0)}
            
            # 推理
            outputs = self.session.run([self.output_name], input_data)[0]
            
            # 后处理
            return self._postprocess(outputs)
        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            raise

    def _postprocess(self, logits: np.ndarray) -> Dict[str, Any]:
        """处理模型输出"""
        # Softmax计算
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # 取Top5结果
        top5_indices = np.argsort(probs[0])[::-1][:5]
        top5_probs = probs[0][top5_indices] * 100
        
        return {
            "results": [
                {
                    "character": self.labels.get(str(idx), "未知"),
                    "probability": f"{prob:.2f}%"
                }
                for idx, prob in zip(top5_indices, top5_probs)
            ]
        }

def find_images(folder: str) -> List[str]:
    """递归查找文件夹中的图像文件"""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

if __name__ == '__main__':
    # 示例用法
    classifier = CalligraphyClassifier(
        model_path=r"onnx\FLENet_T0_24.onnx",
        label_path=r"GF_class_indices.json"
    )
    
    test_images = find_images(r"C:\Users\15503\Desktop\HUNet\CAM\宝")  # 待测试图片
    logger.info(f"找到 {len(test_images)} 张测试图片")
    
    for img_path in test_images:
        try:
            result = classifier.predict(img_path)
            top_char = result['results'][0]['character']
            print(f"{os.path.basename(img_path)} 识别结果: {top_char}")
        except Exception as e:
            logger.warning(f"处理 {img_path} 失败: {str(e)}")


