import numpy as np
import json
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import sys
sys.argv.append("--robot=aloha")
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM
from prismatic.vla.constants import PROPRIO_DIM
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,

)
import cv2
import base64

from dataclasses import dataclass
from typing import Union

import numpy as np

from io import BytesIO
from PIL import Image

@dataclass
class DummyConfig:
    # 只保留必要字段
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/new_data/ckpt/MaskVLA/sim/hanging_mug/loss_siglip_dino_0.3_0.2/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0806_hanging_mug_loss_0.3_0.5--30000_chkpt"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_proprio: bool = True
    use_film: bool = True
    num_images_in_input: int = 3
    center_crop: bool = True
    num_open_loop_steps: int = 25
    unnorm_key: str = "my_aloha_sim_hanging_mug"
    use_relative_actions: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 42

# def numpy_img_to_base64_str(img_np):
#     _, buffer = cv2.imencode('.jpg', img_np)
#     img_bytes = buffer.tobytes()
#     base64_str = base64.b64encode(img_bytes).decode('utf-8')
#     return base64_str

# def create_dummy_observation():
#     """构造与你的 HDF5 数据一致的 observation 结构"""
#     dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
#     dummy_qpos = np.random.randn(14).astype(np.float32)

#     observation = {
#         "images": {
#             "front_t": numpy_img_to_base64_str(dummy_image),  # cam2
#             "left_t": numpy_img_to_base64_str(dummy_image),   # cam1
#             "right_t": numpy_img_to_base64_str(dummy_image),  # cam3
#         },
#         "qpos": dummy_qpos,
#     }

#     return observation


def numpy_img_to_base64_str(img: np.ndarray) -> str:
    """将 numpy RGB 图像转成 base64 编码字符串（JPEG 格式）"""
    # img 是 numpy uint8，格式是 HWC RGB
    pil_img = Image.fromarray(img)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return base64_str

def read_image_rgb(path):
    """从本地路径读取图片并转换为 RGB numpy数组"""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片文件: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def create_dummy_observation():
    """从本地图像读取，构造符合格式的 observation"""

    front_img = read_image_rgb("debug_images/test_image/cam2/output_0001.png")  # cam2，替换成你本地图片路径
    left_img = read_image_rgb("debug_images/test_image/cam1/output_0001.png")   # cam1
    right_img = read_image_rgb("debug_images/test_image/cam3/output_0001.png")  # cam3

    dummy_qpos = np.random.randn(14).astype(np.float32)

    observation = {
        "images": {
            "front_t": numpy_img_to_base64_str(front_img),
            "left_t": numpy_img_to_base64_str(left_img),
            "right_t": numpy_img_to_base64_str(right_img),
        },
        "qpos": dummy_qpos,
    }

    return observation


def test_openvla_with_observation():
    cfg = DummyConfig()

    # 模型组件加载
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim) if cfg.use_l1_regression else None
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM) if cfg.use_proprio else None

    # print("=== Action Head 模块结构 ===")
    # print(action_head)

    # # 如果是Transformer，可以递归打印子模块类型
    # def print_attention_modules(module, prefix=""):
    #     for name, child in module.named_children():
    #         if "attention" in name.lower() or "attn" in name.lower():
    #             print(f"{prefix}{name}: {child}")
    #         print_attention_modules(child, prefix + "  ")
    # if action_head is not None:
    #     print_attention_modules(action_head)

    # 创建 dummy observation
    obs = create_dummy_observation()
    instruction = "Pick up the banana and put it in the basket."

    # 推理动作
    actions = get_vla_action(
        cfg,
        vla,
        processor,
        obs,
        instruction,
        action_head=action_head,
        proprio_projector=proprio_projector,
        use_film=cfg.use_film,
    )


    print("\n=== ✅ OpenVLA 推理结果 ===")
    print("输出动作形状:", [a.shape for a in actions])
    print("输出动作值（第一个）:", actions[0])
    # with open("vla_model_structure.txt", "w", encoding="utf-8") as f:
    #     print(vla, file=f)

    # def print_attention_modules(module, prefix=""):
    #     for name, child in module.named_children():
    #         if "attention" in name.lower() or "attn" in name.lower():
    #             print(f"{prefix}{name}: {child}")
    #         print_attention_modules(child, prefix + "  ")

    # def print_module_names(module, prefix="", max_depth=3):
    #     if max_depth == 0:
    #         return
    #     for name, child in module.named_children():
    #         print(f"{prefix}{name}: {child.__class__.__name__}")
    #         print_module_names(child, prefix + "  ", max_depth - 1)

    # print("LLM 类型:", type(vla.language_model))
    # print("LLM 内部 model:", type(vla.language_model.model))
 


if __name__ == "__main__":
    test_openvla_with_observation()

'''
语言模型的输入改成了自定义 embedding + 自定义 attention mask(没有因果遮挡)，使得原本自回归的 LLaMA 语言模型在推理时变成了类似编码器的双向注意力结构。

因此你模型整体实际上是双向注意力 + 并行预测。

修改的其实是推理时的注意力掩码和输入方式，不是底层权重结构。权重本质相同，掩码改变，推理行为就从单向变成双向，架构表现形式变了，但底层权重完全复用。

'''