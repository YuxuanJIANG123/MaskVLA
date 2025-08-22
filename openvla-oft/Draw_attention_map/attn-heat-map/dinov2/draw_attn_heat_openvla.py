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

# 自动添加命令行参数 --robot=aloha
if "--robot=aloha" not in sys.argv:
    sys.argv.append("--robot=aloha")
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import PROPRIO_DIM
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from models.vision_transformer import vit_small, vit_large
from torchvision import transforms as pth_transforms  # 导入预处理模块
import torch.nn as nn  # 导入神经网络模块

def extract_dinov2_weights(vla_model):
    """
    从VLA模型中提取DINOv2部分的权重
    
    返回:
        state_dict: 包含DINOv2权重的字典
        config: 包含模型配置信息的字典
    """
    dinov2_model = vla_model.vision_backbone.featurizer
    
    # 1. 提取完整的state_dict
    state_dict = dinov2_model.state_dict()
    
    # 2. 提取模型配置信息
    config = {
        'embed_dim': getattr(dinov2_model, 'embed_dim', None),
        'num_heads': getattr(dinov2_model, 'num_heads', None),
        'num_layers': len(dinov2_model.blocks) if hasattr(dinov2_model, 'blocks') else None,
        'patch_size': getattr(dinov2_model.patch_embed, 'patch_size', None),
        'img_size': getattr(dinov2_model.patch_embed, 'img_size', None)
    }
    
    # 3. 保存重要的子模块权重
    important_weights = {
        'patch_embed': {k: v for k, v in state_dict.items() if 'patch_embed' in k},
        'cls_token': state_dict.get('cls_token', None),
        'pos_embed': state_dict.get('pos_embed', None),
        'blocks': {k: v for k, v in state_dict.items() if 'blocks' in k},
        'norm': {k: v for k, v in state_dict.items() if 'norm' in k}
    }
    
    return {
        'full_state_dict': state_dict,
        'config': config,
        'important_weights': important_weights
    }

def save_dinov2_weights(vla_model, save_dir="dinov2_weights"):
    """
    提取并保存DINOv2部分的权重
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取权重
    weights = extract_dinov2_weights(vla_model)
    
    # 保存完整state_dict
    torch.save(weights['full_state_dict'], os.path.join(save_dir, 'full_state_dict.pth'))
    
    # 保存配置信息
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(weights['config'], f, indent=2)
    
    # 保存重要权重
    torch.save(weights['important_weights'], os.path.join(save_dir, 'important_weights.pth'))
    
    print(f"DINOv2 weights saved to {save_dir}")

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


def load_vla_and_save_weights(cfg):
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # 获取 vla 的所有方法和属性
    # methods_and_attrs = dir(vla)

    # # 过滤掉私有方法（以 _ 开头的）
    # public_methods = [name for name in methods_and_attrs if not name.startswith("_")]

    # print("Public methods of `vla`:")
    # for method in public_methods:
    #     print(method)

    # 提取DINOv2权重
    # dinov2_weights = extract_dinov2_weights(vla)
    save_dir="/home/Better-oft/openvla-oft/openvla_extracted_dinov2_weights/robotwin_baseline"
    save_dinov2_weights(vla, save_dir)
    
    # 初始化DINOv2模型
    model = vla.vision_backbone.featurizer
       
    # # 冻结参数
    # for p in model.parameters():
    #     p.requires_grad = False
    
    # model.to(device)
    # model.eval()
    # model.load_state_dict(torch.load(model_path), strict=False)
    # 验证提取的权重
    print("\n验证提取的权重:")
    print(f"模型嵌入维度: {model.embed_dim}")
    print(f"块数量: {len(model.blocks)}")
    print(f"注意力头数: {model.blocks[0].attn.num_heads}")
    """
    模型嵌入维度: 1024
    块数量: 24
    注意力头数: 16
    """
def draw_attn_map():
    #copying a param with shape torch.Size([1, 256, 1024]) from checkpoint, the shape in current model is torch.Size([1, 257, 1024]).
    image_size = (224, 224) # 设置图像大小
    output_dir = '/home/Better-oft/openvla-oft/Draw_attention_map/dino_visual_attn/dinov2'  # 设置输出目录
    patch_size = 14  # 设置patch的大小

    model = vit_large(
        patch_size=14,
        img_size=224,
        init_values=1.0,
        # ffn_layer="mlp",  # 可以选择前馈层的类型，这里被注释掉了
        block_chunks=0
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path="/home/Better-oft/openvla-oft/openvla_extracted_dinov2_weights/robotwin_baseline/full_state_dict.pth"
    model=load_dinov2_weights(model, model_path)
    for p in model.parameters():
        p.requires_grad = False  # 冻结模型参数，不进行梯度更新
    model.to(device)  # 将模型移动到指定的设备
    model.eval()  # 设置模型为评估模式
    print(hasattr(model, 'get_last_self_attention'))  # 输出 True
        # 加载并处理图像
    img = Image.open('/home/Better-oft/openvla-oft/Draw_attention_map/robotwin_image/camera_high/output_0001.png')  # 打开图像文件
    img = img.convert('RGB')  # 转换图像为RGB格式
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),  # 重设图像大小
        pth_transforms.ToTensor(),  # 将图像转换为Tensor
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 归一化处理
    ])
    img = transform(img)  # 应用变换
    print(img.shape)

    # 使图像尺寸适配patch大小
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)  # 调整图像尺寸并增加一个批次维度

    # 计算特征图的宽度和高度
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    print(img.shape)

    # 获取模型最后一层的注意力分数
    attentions = model.get_last_self_attention(img.to(device))

    nh = attentions.shape[1]  # 获取头部数量
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  # 重塑注意力分数
    print(torch.max(attentions, dim=1))  # 打印最大注意力值
    # attentions[:, 283] = 0  # 将特定像素的注意力值设为0

    attentions = attentions.reshape(nh, w_featmap, h_featmap)  # 重塑注意力图
    #attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()  # 上采样注意力图并转为numpy数组

    # attentions = F.interpolate(
    #     attentions.unsqueeze(0), 
    #     scale_factor=patch_size, 
    #     mode="nearest"
    # )[0].cpu().numpy()

    # 双线性插值（适用于平滑过渡）
    attentions = F.interpolate(
        attentions.unsqueeze(0),
        scale_factor=patch_size,
        mode="bilinear",
        align_corners=False  # 是否对齐角点（影响边缘效果）
    )[0].cpu().numpy()

    # # 双三次插值（更高精度，但计算量更大）
    # attentions = F.interpolate(
    #     attentions.unsqueeze(0),
    #     scale_factor=patch_size,
    #     mode="bicubic",
    #     align_corners=False
    # )[0].cpu().numpy()

    # 保存注意力热图
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")  # 设置文件名
        plt.imsave(fname=fname, arr=attentions[j], format='png')  # 保存热图
        print(f"{fname} saved.")  # 打印保存信息



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
        pretrained_checkpoint="/new_data/ckpt/openvla-oft/sim/stack_bowls_three/baseline/openvla-7b+my_aloha_sim_stack_bowls_three+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0812_my_aloha_sim_stack_bowls_three_baseline--30000_chkpt",  # 替换为你的checkpoint路径
        unnorm_key="my_aloha_sim_stack_bowls_three"
    )
    #load_vla_and_save_weights(cfg)
    draw_attn_map()
    

