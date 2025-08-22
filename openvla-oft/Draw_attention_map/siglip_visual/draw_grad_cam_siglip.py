import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import (
    GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM,
    LayerCAM, FullGrad
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from open_clip import create_model_from_pretrained, get_tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# Environment configurations
os.environ['HF_HUB_CACHE'] = '/new_data/hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TORCH_HOME'] = '/new_data/torch_cache'

class Config:
    def __init__(self):
        # Default configurations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_path = '/home/Better-oft/openvla-oft/Draw_attention_map/My_image/pipeline-cat-chonk.jpeg'
        self.labels = ["a Pallas cat", "a lion", "a Siberian tiger"]  # 要评估的标签
        self.method = 'gradcam'  # 可选的CAM方法
        self.output_dir = '/home/Better-oft/openvla-oft/Draw_attention_map/My_image'
        self.model_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
        
        print(f'Using device "{self.device}" for acceleration')

def preprocess_image(preprocess,image_path, device):
    """修正后的预处理函数，确保输出张量可计算梯度"""
    image = Image.open(image_path).convert('RGB')
    
    # 预处理并保留梯度计算能力
    img_tensor = preprocess(image).unsqueeze(0)
    img_tensor = img_tensor.to(device).requires_grad_(True)  # 关键修改
    
    return img_tensor, np.array(image)

def reshape_transform(tensor):
    # 获取输入张量的形状 [batch, num_tokens, dim]
    # print(f"[DEBUG] Reshape transform input shape: {tensor.shape}")
    #[DEBUG] Reshape transform input shape: torch.Size([1, 729, 1152])
    """
    适配 SigLIP ViT-SO400M-14-SigLIP-384 的输出形状 [1, 729, 1152]
    输入说明：
    - tensor.shape = [batch, 27x27, dim] （已不含 [CLS] token）
    - 输出需转为 [batch, dim, 27, 27]
    """
    batch_size, num_tokens, dim = tensor.shape
    grid_size = int(np.sqrt(num_tokens))  # 27
    
    # 直接reshape为空间网格
    features = tensor.reshape(batch_size, grid_size, grid_size, dim)
    # 调整为 [batch, dim, height, width]
    return features.permute(0, 3, 1, 2)

if __name__ == '__main__':
    config = Config()
    
    # 加载SigLIP模型
    model, preprocess = create_model_from_pretrained(config.model_name)
    
    model = model.to(config.device).train()
    tokenizer = get_tokenizer(config.model_name)

    model.zero_grad()
    for param in model.parameters():
        param.requires_grad_(True)  # 确保模型参数可计算梯度
    #print(model)
    # 准备文本输入
    text = tokenizer(config.labels, context_length=model.context_length).to(config.device)
    
    # 加载和预处理图像
    input_tensor, original_image = preprocess_image(preprocess,config.image_path, config.device)
    rgb_img = cv2.resize(original_image, (384, 384))
    rgb_img = np.float32(rgb_img) / 255
    
    # 选择目标层 - SigLIP的最后一个transformer block的LayerNorm前
    target_layers = [model.visual.trunk.blocks[-1].norm1]
    
    # 定义目标函数 - 使用最高概率的标签
    # def text_target_fn(output):
    #     with torch.cuda.amp.autocast():
    #         text_features = model.encode_text(text)
    #         text_features = F.normalize(text_features, dim=-1)
    #         return (output @ text_features.T * model.logit_scale.exp() + model.logit_bias).mean()

    def text_target_fn(output):
        text_features = model.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)
        logits = (output @ text_features.T * model.logit_scale.exp() + model.logit_bias)
        # 比如只选择第一个标签
        return logits[:, 0]

    # 可用的CAM方法
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }
    
    # 计算CAM
    cam_algorithm = methods[config.method]
    with cam_algorithm(model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform) as cam:
        
        # 计算注意力图
        grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[text_target_fn],
                        aug_smooth=True,
                        eigen_smooth=True)
        
        # 取第一个batch和类别
        grayscale_cam = grayscale_cam[0, :]
        
        # 可视化CAM热力图
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 1. 保存CAM热力图
        cam_output_path = os.path.join(config.output_dir, f'siglip_{config.method}_cam.jpg')
        cv2.imwrite(cam_output_path, cam_image)
        print(f"Saved CAM visualization to {cam_output_path}")
            # 2. 计算并保存导向反向传播图（修改版）
        class SigLIPWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                # SigLIP返回的是元组，我们只需要图像特征部分
                return self.model.encode_image(x)
        # 2. 计算并保存导向反向传播图
        gb_model = GuidedBackpropReLUModel(model=SigLIPWrapper(model), device=config.device)
        gb = gb_model(input_tensor, target_category=None)
        gb = deprocess_image(gb)
        
        gb_output_path = os.path.join(config.output_dir, f'siglip_{config.method}_gb.jpg')
        cv2.imwrite(gb_output_path, gb)
        print(f"Saved Guided Backpropagation to {gb_output_path}")
        
        # 3. 计算并保存CAM-GB融合图
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        
        cam_gb_output_path = os.path.join(config.output_dir, f'siglip_{config.method}_cam_gb.jpg')
        cv2.imwrite(cam_gb_output_path, cam_gb)
        print(f"Saved CAM-GB fusion to {cam_gb_output_path}")
    
    # 打印预测概率
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(input_tensor)
        text_features = model.encode_text(text)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        text_probs = torch.sigmoid(image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias)
    
    zipped_list = list(zip(config.labels, [round(p.item(), 3) for p in text_probs[0]]))
    print("Label probabilities: ", zipped_list)