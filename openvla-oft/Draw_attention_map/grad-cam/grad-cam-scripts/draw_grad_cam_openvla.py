import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import base64
import io
import json
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image
import sys
import argparse
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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,SoftmaxOutputTarget 
# 自动添加命令行参数 --robot=aloha
if "--robot=aloha" not in sys.argv:
    sys.argv.append("--robot=aloha")
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_process_output,
)
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import PROPRIO_DIM
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
import torch.nn.functional as F

q01 = np.array([
    0.2744716703891754, -1.985461950302124, 0.9433814173936844, 1.2692148685455322,
    -1.1893224716186523, -2.2242655754089355, 0.00285287294536829, -0.9711185097694397,
    -2.1421759128570557, 0.6029220819473267, -2.561417579650879, -0.15635919854044913,
    -0.5951438176631927, 0.002416828414425254
])
q99 = np.array([
    0.6857023239135742, -0.6628137826919556, 1.9140535593032837, 1.9806477057933793,
    -0.09011215187609795, -0.5776722431182866, 0.05541296083480115, 0.11272602528333664,
    -0.5163271427154541, 2.0033187866210938, 0.22227054461833617, 1.3979896306991573,
    2.985236883163452, 0.061144329607486725
])



def image_to_base64(image):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(image)
        else:
            print(f"Warning: Invalid image shape {image.shape}")
            return None
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def resize_and_center_crop(img: np.ndarray, resize_size=224, crop_scale=0.9) -> np.ndarray:
    """
    Resize and center crop image using OpenCV and NumPy (pure Python version).
    This mimics the model's preprocessing pipeline.

    Args:
        img: BGR image in shape [H, W, 3]
        resize_size: target resize (int or tuple)
        crop_scale: fraction to crop from center after resizing

    Returns:
        Cropped and resized image (uint8, shape [h_crop, w_crop, 3])
    """
    # Resize
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    resized = cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)

    # Center crop
    h, w, _ = resized.shape
    crop_h = int(h * crop_scale)
    crop_w = int(w * crop_scale)
    offset_y = (h - crop_h) // 2
    offset_x = (w - crop_w) // 2
    cropped = resized[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w]

    return cropped


def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result
                
# 创建模型包装器
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, processor, prompt):
        super().__init__()
        self.model = model
        self.processor = processor
        self.prompt = prompt
    
    def forward(self, x_pil):
        # 直接使用PIL图像作为输入
        processed = self.processor(self.prompt, x_pil)
        return self.model(processed)
    
def run_inference_on_images(cfg, base_dir="debug_images/test_image"):
    # === 初始化模型 ===
    vla = get_vla(cfg)
    # 选择目标层 - 通常是最后一个卷积层或Transformer的norm层
    target_layers = [vla.vision_backbone.featurizer.norm]
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim) if cfg.use_l1_regression else None
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM) if cfg.use_proprio else None
    

    
    cam_dirs = {
        'left': os.path.join(base_dir, "cam1"),
        'front': os.path.join(base_dir, "cam2"),
        'right': os.path.join(base_dir, "cam3"),
    }

    image_files = sorted(os.listdir(cam_dirs['front']))
    assert image_files, "No images found in cam2 directory"

    instruction = "Pick up the banana and put it in the basket."  # 将指令定义移到循环外部

    for step, image_filename in enumerate(image_files):
        image_dict = {}
        rgb_images = {}  # 保存原始RGB图像用于可视化
        for view, folder in cam_dirs.items():
            img_path = os.path.join(folder, image_filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")
            
            image_pil = Image.open(img_path).convert("RGB")
            img_b64 = image_to_base64(image_pil)
            image_dict[f"{view}_t"] = img_b64
            # 保存原始图像用于Grad-CAM可视化
            rgb_images[view] = img_b64

        random_qpos = np.random.uniform(low=q01, high=q99)
        observation = {
            "qpos": random_qpos,
            "images": image_dict,
            "timestamp": f"step_{step}"
        }
        

        try:
            # 获取动作并计算梯度
            action = get_vla_action(cfg, vla, processor, observation, instruction,
                                    action_head=action_head,
                                    proprio_projector=proprio_projector,
                                    use_film=cfg.use_film,
                                    return_tensor=True)
            print(f"[DEBUG] Step {step}, action: {action}")

            inputs = get_process_output(cfg, vla, processor, observation, instruction)
            input_tensor = inputs["pixel_values"].float()  # 必须是 float32
            input_tensor.requires_grad_(True)  # 启用梯度
            print(inputs["pixel_values"].dtype)  # 应为 torch.float32
            # 检查 inputs 的键和数据类型
            #print({k: v.dtype for k, v in inputs.items()})  # {'input_ids': torch.int64, 'attention_mask': torch.int64, 'pixel_values': torch.bfloat16}
            
            class ActionPredictionTarget:
                def __init__(self, action_dim):
                    self.action_dim = action_dim

                def __call__(self, model_output):
                    # 确保返回的是浮点张量
                    return model_output[:, self.action_dim].float()  # 强制转换为 float
                
            # 临时包装模型，屏蔽文本输入的梯度
            class VisualOnlyWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, pixel_values):
                    # 固定文本输入（避免触发 nn.Embedding 的梯度计算）
                    text_inputs = {
                        "input_ids": torch.tensor([[0, 1, 2]], device=pixel_values.device),  # 虚拟文本
                        "attention_mask": torch.tensor([[1, 1, 1]], device=pixel_values.device)
                    }
                    return self.model(pixel_values=pixel_values, **text_inputs)
            
            wrapped_model = VisualOnlyWrapper(vla)
                # 初始化Grad-CAM
            cam = GradCAM(model=wrapped_model, 
                        target_layers=target_layers, 
                        reshape_transform=reshape_transform)
            
            # 使用示例
            my_target = [ActionPredictionTarget(action_dim=0)]  # 可视化第1个动作维度

            # 计算Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=my_target)
            print(f"Grayscale CAM shape: {grayscale_cam.shape}")  # grayscale_cam.shape: (1, 14, 14)
            
            # # 确保输入图像是float32类型且在[0,1]范围内
            # rgb_image_float = rgb_images[view].astype(np.float32)
            # if rgb_image_float.max() > 1.0:
            #     rgb_image_float /= 255.0

            # # 可视化
            # visualization = show_cam_on_image(rgb_image_float, grayscale_cam[0], use_rgb=True)
            # # 确保可视化结果是uint8类型
            # if visualization.dtype != np.uint8:
            #     visualization = (visualization * 255).astype(np.uint8)

            # # 保存结果
            # save_path = f"/home/jyx/openvla-oft/Draw_attention_map/My_image/gradcam_{view}_{step}.png"
            # cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                
        except Exception as e:
            logging.error(f"Error in step {step} with {image_filename}: {e}")
            continue



@dataclass
class DeployConfig:
    pretrained_checkpoint: Union[str, Path] = ""
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 3
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 25
    unnorm_key: Union[str, Path] = ""
    use_relative_actions: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 7


if __name__ == "__main__":
    cfg = DeployConfig(
        pretrained_checkpoint="/data/ckpt/openvla-oft/aloha_pick_banana_new/50000steps/openvla-7b+my_aloha_picking_banana_new+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--408",  # 替换为你的checkpoint路径
        unnorm_key="my_aloha_picking_banana_new"
    )
    run_inference_on_images(cfg)
