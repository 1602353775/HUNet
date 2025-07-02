
# 🏛️ HUNet: Hierarchical Universal Network for Multi-Type Ancient Chinese Character Recognition

<div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 15px; border-radius: 8px; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  <div style="display: inline-block; background-color: #fff; padding: 5px 15px; border-radius: 20px; font-size: 14px; color: #555; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <i class="fas fa-history" style="margin-right: 5px; color: #6c757d;"></i> 
    <strong>Evolution Note:</strong> Previously known as <code style="background: #f8f9fa; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', monospace;">FLENet</code>
  </div>
</div>

<div style="display: flex; align-items: center; justify-content: center; margin: 25px 0;">
  <div style="flex: 1; text-align: center;">
    <div style="display: inline-block; position: relative;">
      <span style="position: absolute; top: -15px; left: -15px; background: #ff7043; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-size: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        NEW
      </span>
      <span style="font-size: 28px; font-weight: bold; color: #2c3e50; text-shadow: 1px 1px 3px rgba(0,0,0,0.1);">HUNet</span>
    </div>
    <div style="font-size: 16px; color: #7f8c8d; margin-top: 5px;">Current Version</div>
  </div>
  
  <div style="width: 40px; text-align: center;">
    <i class="fas fa-arrow-right" style="color: #95a5a6;"></i>
  </div>
  
  <div style="flex: 1; text-align: center;">
    <div style="opacity: 0.7;">
      <span style="font-size: 24px; font-weight: bold; color: #95a5a6; font-family: 'Courier New', monospace;">FLENet</span>
      <div style="font-size: 14px; color: #bdc3c7; margin-top: 5px;">Legacy Version</div>
    </div>
  </div>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🌟 ​​Key Features and Core Concepts​


<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
    .enhanced-image {
        max-width: 40%;
        height: auto;
        border: 1px solid #eee;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        filter: brightness(1.05) contrast(1.05);
    }
    .enhanced-image:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .tagline {
        margin-top: 18px;
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 26px;
        letter-spacing: 1.2px;
        color: #2c3e50;
        text-transform: uppercase;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
    }
    .tagline-item {
        display: flex;
        align-items: center;
        padding: 0 10px;
        transition: transform 0.3s ease;
    }
    .tagline-item:hover {
        transform: translateY(-3px);
    }
    .icon-fast {
        margin-right: 8px;
        font-size: 22px;
        color: #FF5722; /* 活力橙 */
    }
    .icon-lean {
        margin-right: 8px;
        font-size: 22px;
        color: #4CAF50; /* 自然绿 */
    }
    .icon-efficient {
        margin-right: 8px;
        font-size: 22px;
        color: #2196F3; /* 科技蓝 */
    }
    .divider {
        color: #bdc3c7;
        font-weight: 300;
    }
</style>

<div style="text-align: center; margin: 20px 0;">
    <img src="assets\bee.jpg" alt="性能对比图" class="enhanced-image"/>
    <div class="tagline">
        <div class="tagline-item">
            <span class="icon-fast"><i class="fas fa-bolt"></i></span>
            <span>Fast</span>
        </div>
        <span class="divider">|</span>
        <div class="tagline-item">
            <span class="icon-lean"><i class="fas fa-leaf"></i></span>
            <span>Lean</span>
        </div>
        <span class="divider">|</span>
        <div class="tagline-item">
            <span class="icon-efficient"><i class="fas fa-cogs"></i></span>
            <span>Efficient</span>
        </div>
    </div>
</div>
<div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 30px 0;"> <div style="flex: 1; min-width: 200px; text-align: center; padding: 15px;"> <i class="fas fa-layer-group" style="font-size: 32px; color: #4CAF50;"></i> <h3>Hierarchical Structure</h3> <p>Multi-level recognition for diverse ancient scripts</p> </div> <div style="flex: 1; min-width: 200px; text-align: center; padding: 15px;"> <i class="fas fa-universal-access" style="font-size: 32px; color: #2196F3;"></i> <h3>Universal Model</h3> <p>Single model handles multiple character types</p> </div> <div style="flex: 1; min-width: 200px; text-align: center; padding: 15px;"> <i class="fas fa-brain" style="font-size: 32px; color: #FF5722;"></i> <h3>Advanced Architecture</h3> <p>Innovative deep learning techniques</p> </div> </div>

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

| Parameter        | Type  | Default                          | Options               | Description                          |
|------------------|-------|----------------------------------|-----------------------|--------------------------------------|
| `--img_size`     | str   | "m"                              | ['s', 'm', 'l', 'n']  | Input image size specification       |
| `--num_classes`  | int   | 8105                             | -                     | Total number of character classes    |
| `--epochs`       | int   | 20                               | -                     | Total training epochs               |
| `--batch-size`   | int   | 256                              | -                     | Number of samples per training batch |
| `--lr`           | float | 0.001                            | -                     | Initial learning rate               |
| `--data-path`    | str   | './datasets'                     | -                     | Root path of training dataset       |
| `--weights`      | str   | "" (empty string)                | -                     | Path to pretrained weights          |
| `--checkpoint`   | str   | "./experiments/best_val_checkpoint/" | -              | Path to save model checkpoints      |
| `--freeze-layers`| flag  | False                            | -                     | Whether to freeze partial network layers |
| `--device`       | str   | 'cuda'                           | ['cuda', 'cpu']       | Training device selection           |


## 🔍 Inference (ONNX Runtime)

```bash
python test.py
```

> 💡 ​​Prerequisite​​: Install ONNX Runtime
```bash
pip install onnxruntime
```

📥 Download ONNX Model：[link](https://pan.baidu.com/s/128r532vfGq4XkxrJoaKb3w?pwd=59u8)


## 💾 Checkpoints

best_val_checkpoint 下载：[link](https://pan.baidu.com/s/1SPgGAD6snK1vWuFlD3EZXA?pwd=tmdp)

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
| Key | Description |
|-----|-------------|
| `epoch` | Current training epoch number |
| `model_state_dict` | Model's parameter dictionary |
| `optimizer_state_dict` | Optimizer's state dictionary |
| `scheduler_state_dict` | Learning rate scheduler's state |
| `best_metric` | Best validation accuracy achieved |
| `model_name` | Name/identifier of the model |

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
        <!-- 缩小后的第一张图（250px） -->
        <div style="display: flex; align-items: center; justify-content: center; height: 300px;">
            <img src="assets\SCAN.jpg" alt="mini application" style="max-height: 250px; width: auto; border: 1px solid #eee; border-radius: 4px; margin: 0 10px; object-fit: contain;"/>
        </div>
        <!-- 原始尺寸的第二张图（300px） -->
        <div style="display: flex; align-items: center; justify-content: center; height: 300px;">
            <img src="assets\APP.png" alt="scan feature" style="max-height: 300px; width: auto; border: 1px solid #eee; border-radius: 4px; margin: 0 10px; object-fit: contain;"/>
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