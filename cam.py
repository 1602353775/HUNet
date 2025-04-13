import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models
from models.HUNet import HUNet_24
from models.CNNs.mobilenet_v3 import mobilenet_v3_small
from models.CNNs.fasternet import FasterNet_T0
from models.Shift.shiftvit import ShiftViT_T0
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Torch device to use')
    parser.add_argument(
        '--image-dir',
        type=str,
        default='./CAM/input',
        help='Input image directory')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='fem',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'fullgrad',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam', 'shapleycam',
                            'finercam'
                        ],
                        help='CAM method')
    parser.add_argument('--output-dir', type=str, default='./CAM/output',
                        help='Root output directory')
    args = parser.parse_args()
    
    print(f'Using device "{args.device}" for acceleration' if args.device == 'cuda' else 'Using CPU for computation')
    return args

def initialize_models(device):
    """Initialize models with checkpoints"""
    models_dict = {}
    
    # Initialize HUNet
    hunet = HUNet_24(num_classes=8105).to(device)
    hunet_checkpoint = torch.load(
        r'experiments\best_val_checkpoint\HUNet_24\best_val_checkpoint.pt',
        map_location=device
    )
    hunet.load_state_dict(hunet_checkpoint['model_state_dict'], strict=False)
    hunet.eval()
    
    # Initialize MobileNetV3
    mobilenet = mobilenet_v3_small(num_classes=8105).to(device)
    mobilenet_checkpoint = torch.load(
        r'experiments\best_val_checkpoint\mobilenet_v3_small\best_val_checkpoint.pt',
        map_location=device
    )
    mobilenet.load_state_dict(mobilenet_checkpoint['model_state_dict'], strict=False)
    mobilenet.eval()


    fastrnet =  FasterNet_T0().to(device)
    weights_dict = torch.load(r'experiments\best_val_checkpoint\FasterNet_T0\best_val_model.pth', map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                            if fastrnet.state_dict()[k].numel() == v.numel()}
    fastrnet.load_state_dict(load_weights_dict, strict=False)
    fastrnet.eval()
    
    shfitvit = ShiftViT_T0().to(device)
    shfitvit_checkpoint = torch.load(
        r'experiments\best_val_checkpoint\ShiftViT_T0\best_val_checkpoint.pt',
        map_location=device
    )
    shfitvit.load_state_dict(shfitvit_checkpoint['model_state_dict'], strict=False)
    shfitvit.eval()
    
    models_dict = {
        "HUNet": {
            "model": hunet,
            "target_layers": [hunet.local_layers[2].blocks[1].spatial_mixing.partial_conv3],
            "target_layers": [hunet.local_layers[2].blocks[1].spatial_mixing.partial_conv3,
                             hunet.interactive_layers[1].reduction,
                             hunet.global_layers[2].blocks[1].mlp.fc2]
        },
        "MobileNetV3": {
            "model": mobilenet,
            "target_layers": [mobilenet.features[4].block[3][0],
                            mobilenet.features[8].block[3][0],
                            mobilenet.features[-3]]
        },
         "FasterNet_T0": {
            "model": fastrnet,
            "target_layers": [fastrnet.stages[6].blocks[1].spatial_mixing.partial_conv3]
        },
         "ShiftViT_T0": {
            "model": shfitvit,
            "target_layers": [shfitvit.layers[3].blocks[1].mlp.fc2 ]
        }

    }
    return models_dict

def process_image(image_path, device, target_size=128):
    """Process single image with denoising and padding"""
    img = Image.open(image_path).convert('RGB')
    
    # Preprocessing pipeline
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, ksize=5)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    
    # Resize and pad
    h, w = denoised_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(denoised_rgb, new_size, interpolation=cv2.INTER_NEAREST)
    padded = cv2.copyMakeBorder(resized, 
                               (target_size - new_size[1])//2, 
                               (target_size - new_size[1]+1)//2,
                               (target_size - new_size[0])//2, 
                               (target_size - new_size[0]+1)//2,
                               cv2.BORDER_REPLICATE)
    
    # Prepare input tensor
    rgb_img = np.float32(padded) / 255
    input_tensor = preprocess_image(rgb_img, 
                                   mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5]).to(device)
    return input_tensor, rgb_img, os.path.splitext(os.path.basename(image_path))[0]

def generate_cam(cam_method, model, target_layers, input_tensor, rgb_img, device, aug_smooth, eigen_smooth):
    """Generate CAM visualizations"""
    with cam_method(model=model, target_layers=target_layers) as cam:
        cam.device = device
        grayscale_cam = cam(input_tensor=input_tensor,
                           targets=None,
                           aug_smooth=aug_smooth,
                           eigen_smooth=eigen_smooth)[0, :]
                           
        # Generate CAM overlay
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
        # Generate Guided Backprop
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
        gb = gb_model(input_tensor, target_category=None)
        
        # Generate CAM-GBP fusion
        cam_mask = cv2.merge([grayscale_cam]*3)
        cam_gb = deprocess_image(cam_mask * gb)
        
        return cam_image, deprocess_image(gb), cam_gb

def save_image(path, img):
    """通用保存函数，自动处理中文路径问题"""
    try:
        # 尝试用OpenCV原生方式保存
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(path)
    except:
        # 回退方案1：使用imencode
        path_encoded = path.encode('utf-8').decode('gbk') if os.name == 'nt' else path
        success, buffer = cv2.imencode('.jpg', img)
        if success:
            with open(path_encoded, 'wb') as f:
                buffer.tofile(f)
       

def main():
    args = get_args()
    device = torch.device(args.device)
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM,
        'finercam': FinerCAM
    }

    
    # Initialize models
    models_dict = initialize_models(device)
    
    # Create output directories
    for model_name in models_dict:
        os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    
    # Process images
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    image_files = [f for f in os.listdir(args.image_dir) 
                  if os.path.splitext(f)[1].lower() in valid_extensions]
    
    for filename in image_files:
        image_path = os.path.join(args.image_dir, filename)
        try:
            input_tensor, rgb_img, base_name = process_image(image_path, device)
            
            for model_name, model_info in models_dict.items():
                # Generate visualizations
                cam_method = methods[args.method]
                cam_img, gb_img, cam_gb_img = generate_cam(
                    cam_method,
                    model_info["model"],
                    model_info["target_layers"],
                    input_tensor,
                    rgb_img,
                    device,
                    args.aug_smooth,
                    args.eigen_smooth
                )
                
                # Save outputs
                output_subdir = os.path.join(args.output_dir, model_name)
                save_image(os.path.join(output_subdir, f"{base_name}_{model_name}_cam.jpg"), cam_img)
                save_image(os.path.join(output_subdir, f"{base_name}_{model_name}_gb.jpg"), gb_img)
                save_image(os.path.join(output_subdir, f"{base_name}_{model_name}_cam_gb.jpg"), cam_gb_img)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

if __name__ == '__main__':
    main()