import os
import cv2
import numpy as np
import torch
from torchvision import models
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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST
from torchvision import transforms

# Environment configurations
os.environ['HF_HUB_CACHE'] = '/new_data/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TORCH_HOME'] = '/new_data/torch_cache'

class Config:
    def __init__(self):
        # Default configurations
        self.device = 'cuda'
        self.image_path = '/home/Better-oft/openvla-oft/debug_images/test_image/cam2/output_0001.png'
        self.aug_smooth = False
        self.eigen_smooth = False
        self.method = 'gradcam'
        self.output_dir = '/home/Better-oft/openvla-oft/Draw_attention_map/GradCAM_output'
        
        # Print device info
        if self.device:
            print(f'Using device "{self.device}" for acceleration')
        else:
            print('Using CPU for computation')

def my_preprocess_image(image_path, device, target_size=224):
    # 读取图像并确保是RGB格式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建预处理流程
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),  # 强制缩放为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 应用预处理
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加batch维度
    return input_tensor

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    config = Config()
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

    model = models.vit_b_16(pretrained=True).to(torch.device(config.device)).eval()

    print(model)

    target_layers = [model.encoder.layers[-1].ln_1]

    image = cv2.imread(config.image_path, 1)[:, :, ::-1]
    resized_img = cv2.resize(image, (224, 224))
    rgb_img = np.float32(resized_img) / 255

    input_tensor = my_preprocess_image(config.image_path, config.device)
    print(f'Input tensor shape: {input_tensor.shape}') #([1, 3, 224, 224])
    
    targets = None

    cam_algorithm = methods[config.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                        reshape_transform=reshape_transform) as cam:

        cam.batch_size = 1
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=config.aug_smooth,
                            eigen_smooth=config.eigen_smooth)
        print(f'Grayscale CAM shape: {grayscale_cam.shape}')
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, device=config.device)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    os.makedirs(config.output_dir, exist_ok=True)

    cam_output_path = os.path.join(config.output_dir, f'{config.method}_cam.jpg')
    gb_output_path = os.path.join(config.output_dir, f'{config.method}_gb.jpg')
    cam_gb_output_path = os.path.join(config.output_dir, f'{config.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    cv2.imwrite(gb_output_path, gb)
    cv2.imwrite(cam_gb_output_path, cam_gb)