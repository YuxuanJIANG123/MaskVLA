import os
import cv2
import numpy as np
import torch

import sys
sys.path.append('/home/Better-oft/openvla-oft/Draw_attention_map/dino_visual_attn/dinov2')

import timm
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
from safetensors.torch import load_file
from models.vision_transformer import vit_small, vit_large

# Environment configurations
os.environ['HF_HUB_CACHE'] = '/new_data/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TORCH_HOME'] = '/new_data/torch_cache'

cam_name = 'gradcam'

image_size = 224  # 图像尺寸
d_height = d_width = image_size // 14  # 计算特征图的高度和宽度

class Config:
    def __init__(self):
        # Default configurations
        self.device = 'cuda'
        self.image_path = '/home/Better-oft/openvla-oft/Draw_attention_map/robotwin_image/camera_high/output_0001.png'
        self.aug_smooth = False
        self.eigen_smooth = False
        self.method = cam_name
        self.output_dir = f'/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/gradcam_dinov2_output/{cam_name}/image1/base'
        
        # Print device info
        if self.device:
            print(f'Using device "{self.device}" for acceleration')
        else:
            print('Using CPU for computation')

def load_dinov2_weights(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    
    # 检查pos_embed形状是否匹配
    if 'pos_embed' in state_dict and state_dict['pos_embed'].shape != model.pos_embed.shape:
        print(f"Adjusting pos_embed shape from {state_dict['pos_embed'].shape} to {model.pos_embed.shape}")
        
        # 获取原始pos_embed (不包括class token的位置编码)
        pos_embed = state_dict['pos_embed']
        
        # 获取class token的位置编码 (通常为全零)
        class_pos_embed = model.pos_embed[:, 0:1, :]
        
        # 拼接class token的位置编码和patch位置编码
        new_pos_embed = torch.cat([class_pos_embed, pos_embed], dim=1)
        state_dict['pos_embed'] = new_pos_embed
    
    # 非严格加载，忽略不匹配的键
    model.load_state_dict(state_dict, strict=False)
    return model

def my_preprocess_image(image_path, device, target_size=image_size):
    # 读取图像并确保是RGB格式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建预处理流程
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),  # 强制缩放为image_sizeximage_size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 应用预处理
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加batch维度
    return input_tensor

def reshape_transform(tensor, height=d_height, width=d_width):
    # 去掉cls token

    print(f"tensor shape: {tensor.shape}, size(1): {tensor.size(1)}")

    num_tokens = tensor.size(1) - 1  # 去掉cls token,1,1373,1024
    
    result = tensor[:, 1:, :].reshape(tensor.size(0),height, height, tensor.size(2))
    # 1,37,37,1373
    # result = tensor[:, 1:, :].reshape(tensor.size(0),
    # height, width, tensor.size(2))

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


    # model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', 
    #                         pretrained=False).to(torch.device(config.device))

    # # 加载safetensors权重
    # state_dict = load_file('/new_data/hf_cache/models--timm--vit_large_patch14_reg4_dinov2.lvd142m/snapshots/ckpt/model.safetensors')
    # model.load_state_dict(state_dict)
    # model.eval()
    model = vit_large(
        patch_size=14,
        img_size=224,
        init_values=1.0,
        # ffn_layer="mlp",  
        block_chunks=0
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path="/home/Better-oft/openvla-oft/openvla_extracted_dinov2_weights/robotwin_baseline/full_state_dict.pth"
    model=load_dinov2_weights(model, model_path)
    model.to('cuda')

    #print(model)



    # target_layers = [model.encoder.layers[-1].ln_1]
    target_layers = [model.blocks[-1].norm1] 

    image = cv2.imread(config.image_path, 1)[:, :, ::-1]
    resized_img = cv2.resize(image, (image_size, image_size))
    rgb_img = np.float32(resized_img) / 255

    input_tensor = my_preprocess_image(config.image_path, config.device)
    print(f'Input tensor shape: {input_tensor.shape}') #([1, 3, image_size, image_size])
    
    output = model(input_tensor)
    print(output.shape)  # 输出形状

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