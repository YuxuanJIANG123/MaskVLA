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

def letterbox_resize(img: np.ndarray, target_size=224, padding_color=(114, 114, 114)) -> np.ndarray:
    """
    Resize image using letterbox (aspect ratio preserving with padding).

    Args:
        img: BGR image in shape [H, W, 3]
        target_size: final output size (int or tuple, square output)
        padding_color: color used for padding (B, G, R)

    Returns:
        Letterbox resized image (uint8, shape [target_size, target_size, 3])
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    target_w, target_h = target_size

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize with aspect ratio preserved
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create new canvas and paste resized image onto center
    canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas

def save_attention_heatmaps_for_three_views(attn, image_paths, save_path):
    """
    保存三张图像的注意力热图，并横向拼接保存。
    
    Args:
        attn (torch.Tensor): 注意力张量，形状 [1, num_heads, T, T] 或 [num_heads, T, T]
        image_paths (List[str]): 三张图路径，顺序为 [cam2, cam1, cam3]
        save_path (str): 最终拼接图保存路径
    """
    if attn.dim() == 4:
        attn = attn[0]  # [num_heads, T, T]

    attn_avg = attn.mean(0)  # [T, T]
    cls_attn = attn_avg[0, 1:]  # [T-1], 去掉 CLS->CLS

    num_views = 3
    num_total_patches = cls_attn.shape[0]
    num_patches_per_view = num_total_patches // num_views
    print(f"[INFO] 总 patch 数量: {num_total_patches}")
    print(f"[INFO] 每张图像的 patch 数量: {num_patches_per_view}")

    assert num_total_patches % num_views == 0, "无法平均分割 attention 为每张图像！"

    overlay_imgs = []

    for i, img_path in enumerate(image_paths):
        orig = cv2.imread(img_path)
        if orig is None:
            print(f"[❌] 无法读取图像: {img_path}")
            return
        
        # 预处理图像：resize 并中心裁剪
        orig = resize_and_center_crop(orig, resize_size=224, crop_scale=0.9)
        # orig =letterbox_resize(orig, target_size=224, padding_color=(114, 114, 114))
        H, W, _ = orig.shape

        # 提取对应视角的 attention
        start = i * num_patches_per_view
        end = (i + 1) * num_patches_per_view
        attn_slice = cls_attn[start:end]  # [N_patch_per_view]

        # 转换为 2D 网格，pad 到正方形
        grid_size = int(np.ceil(num_patches_per_view ** 0.5))
        pad_len = grid_size ** 2 - num_patches_per_view
        attn_map = F.pad(attn_slice, (0, pad_len), value=0).reshape(grid_size, grid_size)
        attn_map = attn_map.cpu().numpy()
        attn_map = attn_map / (attn_map.max() + 1e-8)  # 归一化避免除0

        # resize 到图像大小，最近邻插值
        attn_map_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_NEAREST)
        # attn_map_resized = cv2.resize(attn_map, (W, H))

        # 生成彩色热图
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)

        # 叠加到原图
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        overlay_imgs.append(overlay)

    # 拼接并保存
    combined = np.concatenate(overlay_imgs, axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    success = cv2.imwrite(save_path, combined)

    if success:
        print(f"[✅] 拼接注意力热图保存成功: {save_path}")
    else:
        print(f"[❌] 拼接注意力热图保存失败: {save_path}")
    return combined

def register_vit_attention_hook(vla, layer_index=23):
    """
    注册 vla 模型中第 layer_index 层 ViT Block 的注意力 hook。
    
    Args:
        vla: OpenVLA 模型（已加载）
        layer_index: 想 hook 的 block 层数（0-23）
        
    Returns:
        saved: 包含 'attn' attention 矩阵的字典
        hook: 注册好的 hook，可用于移除
    """
    saved = {}

    def forward_hook(module, input, output):
        # print(f"[HOOK] 🔥 Attention hook triggered! Forward attention from layer {layer_index}")
        if isinstance(output, tuple) or isinstance(output, list):
            attn = output[1]  # 多数模型结构
        else:
            attn = output  # 有些可能直接输出 attn
        attn_probs = F.softmax(attn, dim=-1)
        saved['attn'] = attn_probs.detach().cpu()
        # print("attn shape:", attn_probs.shape)
        # print("min:", attn_probs.min().item(), "max:", attn_probs.max().item())
        # print("example row sum:", attn_probs[0, 0].sum().item())  # 第一层第一个head的权重和

        '''
        attn shape: torch.Size([1, 261, 1024])
        min: -3.093590497970581 max: 11.964150428771973
        example row sum: 30.967235565185547
        '''

    # 安全性检查：防止越界
    blocks = vla.vision_backbone.featurizer.blocks
    assert 0 <= layer_index < len(blocks), f"Invalid layer_index={layer_index}, total layers={len(blocks)}"

    attn_module = blocks[layer_index].attn
    hook = attn_module.register_forward_hook(forward_hook)

    return saved, hook


def run_inference_on_images(cfg, base_dir="debug_images/test_image"):
    # === 初始化模型 ===
    vla = get_vla(cfg)
    
    # # 注册注意力hook
    # saved, forward_hook, backward_hook =register_attention_hook(vla)
    # L = 23
    # attn_module = vla.vision_backbone.featurizer.blocks[L].attn
       
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
    video_frames_dict = {L: [] for L in range(24)}  # 每层的视频帧缓存

    for step, image_filename in enumerate(image_files):
        image_dict = {}
        for view, folder in cam_dirs.items():
            img_path = os.path.join(folder, image_filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")
            image_pil = Image.open(img_path).convert("RGB")
            img_b64 = image_to_base64(image_pil)
            image_dict[f"{view}_t"] = img_b64

        random_qpos = np.random.uniform(low=q01, high=q99)
        # qpos=[0.0]*14
        observation = {
            "qpos": random_qpos,
            "images": image_dict,
            "timestamp": f"step_{step}"
        }

        instruction = "Pick up the banana and put it in the basket."

        try:
            ##画最后一层的attn图
            # #在调用 get_vla_action() 前注册好，它就能在 forward 时捕获本轮的注意力。
            # saved_attn, hook = register_vit_attention_hook(vla, layer_index=23)

            # action = get_vla_action(cfg, vla, processor, observation, instruction,
            #                                 action_head=action_head,
            #                                 proprio_projector=proprio_projector,
            #                                 use_film=cfg.use_film,
            #                                 return_tensor=True)
            # print(f"[DEBUG] Step {step}, action: {action}")

            # image_paths = [
            # os.path.join(cam_dirs['front'], image_filename),  # cam2 前摄像头
            # os.path.join(cam_dirs['right'], image_filename),  # cam1 右腕摄像头
            # os.path.join(cam_dirs['left'], image_filename),   # cam3 左腕摄像头
            # ]

            # # 保存热图
            # save_path = f"/home/jyx/openvla-oft/Draw_attention_map/My_image/attn.png"
            # save_cls_attention_heatmap(saved_attn['attn'], image_paths[0], save_path)
            # hook.remove()  # 清理 hook

            # 注册所有 0~23 层注意力 hook
            saved_attns = {}
            hooks = []
            
            for L in range(24):
                saved, hook = register_vit_attention_hook(vla, layer_index=L)
                saved_attns[L] = saved
                hooks.append(hook)

            # 推理（会触发 forward，从而填充 saved_attns 中的注意力）
            action = get_vla_action(cfg, vla, processor, observation, instruction,
                                    action_head=action_head,
                                    proprio_projector=proprio_projector,
                                    use_film=cfg.use_film,
                                    return_tensor=True)
            # print(f"[DEBUG] Step {step}, action: {action}")

            image_paths = [
                os.path.join(cam_dirs['front'], image_filename),  # cam2 前摄像头
                os.path.join(cam_dirs['right'], image_filename),  # cam1 右腕摄像头
                os.path.join(cam_dirs['left'], image_filename),   # cam3 左腕摄像头
            ]

            # 遍历每一层，画图
            for L in range(27):
                attn = saved_attns[L].get('attn')
                if attn is not None:
                    save_path = f"./Draw_attention_map/My_image/siglip/layer{L:02d}.png"
                    # save_cls_attention_heatmap(attn, image_paths[0], save_path)
                    combined=save_attention_heatmaps_for_three_views(attn, image_paths, save_path)
                    video_frames_dict[L].append(combined)
                else:
                    print(f"[⚠️] No attention saved for layer {L}")
            # save_attention_weight_density(attn, "/home/jyx/openvla-oft/Draw_attention_map/My_image/attn_density.png")
            # 清理所有 hook
            for hook in hooks:
                hook.remove()
        except Exception as e:
            logging.error(f"Error in step {step} with {image_filename}: {e}")
            continue

    video_output_dir = "./Draw_attention_map/My_image/"
    os.makedirs(video_output_dir, exist_ok=True)

    # for L in range(24):
    #     save_path = os.path.join(video_output_dir, f"layer{L:02d}.mp4")
    #     save_video_from_frames(video_frames_dict[L], save_path, fps=10)




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
