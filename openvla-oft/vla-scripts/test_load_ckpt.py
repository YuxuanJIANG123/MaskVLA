import os
os.environ['HF_HUB_CACHE'] = '/new_data/hf_cache'
import json

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
from prismatic.models import load
import os
import json
from prismatic.conf import VLAConfig, VLARegistry
from dataclasses import dataclass, field
from prismatic.overwatch import initialize_overwatch
from prismatic.models.vlms import PrismaticVLM
import torch
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
import sys
from huggingface_hub import HfFileSystem, hf_hub_download
from prismatic.models.materialize import get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from PIL import Image
import numpy as np
import torch.nn.functional as F



# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

hf_token: Union[str, Path] = Path(".hf_token")
pretrained_checkpoint=Path("/new_data/hf_cache/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12")
vla: VLAConfig = field(
    default_factory=VLAConfig.get_choice_class(VLARegistry.FREEZE_SIGLIP_224PX_MX_BRIDGE.vla_id)
)
run_root_dir="/data/ckpt/openvla/train_from_scratch"

def load_vision_backbone_only(model_id_or_path: Union[str, Path], cache_dir: Optional[str] = None, device_id: int = 0):
    """Load only the Vision Backbone."""
    if model_id_or_path not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

    overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
    with overwatch.local_zero_first():
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]
    
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )
    vision_backbone.to(device_id)
    print(f"[INFO!]Vision Backbone load in device {device_id} successfully!")
    return vision_backbone, image_transform

def load_vision_backbone_and_image_transform(device_id) -> PrismaticVLM:

    torch.cuda.empty_cache()

    vision_backbone, image_transform = load_vision_backbone_only("prism-dinosiglip-224px+7b", cache_dir="/new_data/hf_cache",device_id=device_id)
    print("vision_backbone, image_transform 加载成功!")
    # with open('vision_backbone.txt', 'w') as f:
    #     sys.stdout = f  # 重定向标准输出到文件
    #     print(vision_backbone)
    #     sys.stdout = sys.__stdout__  # 恢复标准输出
    # print("vision_backbone 已保存到文件 vision_backbone.txt")
    # with open('image_transform.txt', 'w') as f:
    #     sys.stdout = f  # 重定向标准输出到文件
    #     print(image_transform)
    #     sys.stdout = sys.__stdout__  # 恢复标准输出
    # print("image_transform 已保存到文件 image_transform.txt")
    return vision_backbone, image_transform

def get_gt_feature(vision_backbone, img_regular,img_fused):
    vision_backbone = vision_backbone.to('cuda')
    gt_siglip_feature = vision_backbone.module.siglip_featurizer(img_fused).to('cuda')
    gt_dinov2_feature = vision_backbone.module.dino_featurizer(img_regular).to('cuda')

    return gt_siglip_feature, gt_dinov2_feature
    
def siglip_contrastive_loss(
    text_embeddings,  # [batch_size, dim]
    image_features,   # [batch_size, dim]
    logit_scale,      # 可学习的温度参数
    logit_bias=None,  # SIGLIP 可选的偏置项
    labels=None,      # 自定义标签（默认为对角线）
    reduction='mean'  # 损失 reduction 方式
):
    # 归一化特征
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_features = F.normalize(image_features, p=2, dim=-1)
    batch_size = text_embeddings.shape[0]
    
    # 计算相似度矩阵
    logits_per_text = torch.matmul(text_embeddings, image_features.t()) * logit_scale
    logits_per_image = logits_per_text.t()
    
    # 添加偏置项（如果存在）
    if logit_bias is not None:
        logits_per_text = logits_per_text + logit_bias
        logits_per_image = logits_per_image + logit_bias
    
    # 创建标签（默认为对角线匹配）
    if labels is None:
        labels = torch.arange(batch_size, device=text_embeddings.device)
    
    # 计算对称的对比损失
    loss_text = F.cross_entropy(logits_per_text, labels, reduction=reduction)
    loss_image = F.cross_entropy(logits_per_image, labels, reduction=reduction)
    
    return (loss_text + loss_image) / 2

if __name__ == "__main__":
    device_id = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    vision_backbone, image_transform = load_vision_backbone_and_image_transform(device_id)
    # 假设输入是 4 张图像的批量张量 (4, 3, 224, 224)
    batch_tensor = torch.rand(4, 3, 224, 224).to('cuda')  # 示例数据

    # # 逐张转换为 PIL Image
    # batch_pil_images = []
    # for img_tensor in batch_tensor:  # img_tensor shape: (3, 224, 224)
    #     # 转换为 HWC NumPy (范围 0-1 -> 0-255)
    #     numpy_img = img_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
    #     numpy_img = (numpy_img * 255).astype(np.uint8)  # float [0,1] -> uint8 [0,255]
    #     pil_img = Image.fromarray(numpy_img)
    #     batch_pil_images.append(pil_img)
    gt_siglip_feature, gt_dinov2_feature = get_gt_feature(vision_backbone, image_transform, batch_tensor, batch_tensor)
    print("GT features extracted successfully!")
    print(f"GT Siglip Feature Shape: {gt_siglip_feature.shape}")#torch.Size([4, 256, 1024])
    print(f"GT Dinov2 Feature Shape: {gt_dinov2_feature.shape}")#torch.Size([4, 256, 1152])

