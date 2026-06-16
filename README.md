# 🏛️ HUNet: Hierarchical Universal Network for Multi-Type Ancient Chinese Character Recognition

> 📜 **Evolution Note**: Previously known as `FLENet`

<div align="center">

✨ **HUNet** (Current Version) → ✨ **FLENet** (Legacy Version)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🌟 Key Features and Core Concepts

<div align="center">
  <img src="assets/bee.jpg" alt="Performance Comparison" width="40%">
</div>

### 🚀 Core Advantages

| Feature             | Description                          | Emoji |
| ------------------- | ------------------------------------ | ----- |
| **Fast**      | Optimized for high-speed recognition | ⚡    |
| **Lean**      | Minimal resource consumption         | 🌿    |
| **Efficient** | State-of-the-art accuracy            | 🛠️  |

### 🔍 Key Components

|      | Feature                          | Description                                         |
| ---- | -------------------------------- | --------------------------------------------------- |
| 🏗️ | **Hierarchical Structure** | Multi-level recognition for diverse ancient scripts |
| 🌐   | **Universal Model**        | Single model handles multiple character types       |
| 🧠   | **Advanced Architecture**  | Innovative deep learning techniques                 |

## 📂 Dataset Preparation

### 🗂 Directory Structure

```
datasets/
├── train/                      # Training set
│   ├── 篆书/                   # Seal script
│   │   ├── 爱/                 # Character "爱"  
│   │   │   ├── 1.jpg           # Sample image
│   │   │   ├── 2.jpg
│   │   │   └── ...             # Other samples
│   │   ├── 书/                 # Other characters
│   │   └── ...             
│   │
│   ├── 隶书/                   # Clerical script
│   │   ├── 爱/
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   └── ...             
│   │
│   └── ...                     # Other script types
│
└── test/                       # Test set
    ├── test_1/                 # Test set 1
    │   ├── 爱/
    │   │   ├── 1.jpg
    │   │   └── ...
    │   └── ...             
    │
    └── test_2/                 # Test set 2
        ├── 爱/
        │   ├── 1.jpg
        │   └── ...
        └── ...
```

### 📸 Sample Images

<div style="text-align: center; margin: 20px 0;">
    <img src="assets/dataset.png" alt="性能对比图" style="max-width: 80%; height: auto; border: 1px solid #eee; border-radius: 4px;"/>
</div>

## 📋使用许可要求

## 许可协议申请流程

使用 **MTACCR** 数据集，研究者需要完成以下步骤：

### 📄 下载使用协议

下载《MTACCR数据集使用许可协议（学术研究专用版）》：**datasets\MTACCR数据集使用许可协议.docx**

### ✍️ 填写申请信息

协议中需要填写以下个人信息：

- **姓名及职务/身份**
- **所属单位**（高校/公司全称）
- **联系方式**
- **研究方向及代表性研究成果**
- **研究目的和计划**

### 📝 签署与提交

- 申请人亲笔签名并单位加盖公章
- 协议双面打印后签署
- 扫描件发送至：**qihengnian@foxmail.com**

## 审核与获取

- **审核周期**：7个工作日内完成审核
- **审核通过**：邮件发送数据集访问权限
- **审核未通过**：将说明具体原因

## 🚀 Model Training

```bash
python train.py \
    --img_size "m" \
    --num_classes 8105 \
    --epochs 20 \
    --batch-size 256 \
    --lr 0.001 \
    --data-path "./datasets" \
    --checkpoint "./experiments/best_val_checkpoint/"
```

## ⚙️ Training Parameters

| Parameter           | Type  | Default                              | Options              | Description                              |
| ------------------- | ----- | ------------------------------------ | -------------------- | ---------------------------------------- |
| `--img_size`      | str   | "m"                                  | ['s', 'm', 'l', 'n'] | Input image size specification           |
| `--num_classes`   | int   | 8105                                 | -                    | Total number of character classes        |
| `--epochs`        | int   | 20                                   | -                    | Total training epochs                    |
| `--batch-size`    | int   | 256                                  | -                    | Number of samples per training batch     |
| `--lr`            | float | 0.001                                | -                    | Initial learning rate                    |
| `--data-path`     | str   | './datasets'                         | -                    | Root path of training dataset            |
| `--weights`       | str   | "" (empty string)                    | -                    | Path to pretrained weights               |
| `--checkpoint`    | str   | "./experiments/best_val_checkpoint/" | -                    | Path to save model checkpoints           |
| `--freeze-layers` | flag  | False                                | -                    | Whether to freeze partial network layers |
| `--device`        | str   | 'cuda'                               | ['cuda', 'cpu']      | Training device selection                |

## 🔍 Inference (ONNX Runtime)

```bash
python test.py
```

> 💡 Prerequisite: Install ONNX Runtime

```bash
pip install onnxruntime
```

📥 Download ONNX Model：[link](https://pan.baidu.com/s/1ax6_5mYPGRsO2mvgc20AsQ?pwd=5npy)

## 💾 Checkpoints

best_val_checkpoint 下载：[link](https://pan.baidu.com/s/1jsIReyLQoM-FaHuzuuWWpw?pwd=ukxb)

🏷️ Checkpoint Structure

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_metric': val_acc,
    'model_name': model_name,
}
```

### Checkpoint Structure Explanation:

| Key                      | Description                       |
| ------------------------ | --------------------------------- |
| `epoch`                | Current training epoch number     |
| `model_state_dict`     | Model's parameter dictionary      |
| `optimizer_state_dict` | Optimizer's state dictionary      |
| `scheduler_state_dict` | Learning rate scheduler's state   |
| `best_metric`          | Best validation accuracy achieved |
| `model_name`           | Name/identifier of the model      |

### 💻 Usage Example

```python
# Save checkpoint
torch.save(checkpoint, 'model_checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 📱 Demo Application

<div style="display: flex; justify-content: center; align-items: center; margin: 20px 0; overflow-x: auto;">
    <div style="display: inline-flex; align-items: center; min-width: fit-content;">
        <!-- First image slightly reduced (280px) -->
        <div style="display: flex; align-items: center; justify-content: center; height: 320px;">
            <img src="assets\SCAN.jpg" alt="mini application" style="max-height: 280px; width: auto; border: 1px solid #eee; border-radius: 4px; margin: 0 10px; object-fit: contain;"/>
        </div>
        <!-- Second image reduced to match better (280px) -->
        <div style="display: flex; align-items: center; justify-content: center; height: 320px;">
            <img src="assets\APP.png" alt="scan feature" style="max-height: 280px; width: auto; border: 1px solid #eee; border-radius: 4px; margin: 0 10px; object-fit: contain;"/>
        </div>
    </div>
</div>

### 🙏 Acknowledgments

We would like to express our gratitude to the following projects and resources :

- [AncientGlyphNet](https://github.com/youngbbi/AncientGlyphNet) : an advanced deep learning framework for detecting ancient Chinese characters in complex scene.

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{wang2025hunet,
  title={HUNet: Hierarchical Universal Network for Multi-Type Ancient Chinese Character Recognition},
  author={Wang, Zhaojiang and Zhang, Chu and Lang, Qing and Jin, Lianwen and Qi, Hengnian},
  journal={npj Heritage Science},
  volume={13},
  number={1},
  pages={1--16},
  year={2025},
  publisher={Nature Publishing Group}
}
```

<div align="center" style="margin-top: 40px;"> <p style="color: #666; font-size: 1.1em; font-style: italic;"> <i class="fas fa-monument" style="color: #8B4513;"></i> Preserving Cultural Heritage Through AI <i class="fas fa-robot" style="color: #4B9CD3;"></i> </p> </div>
