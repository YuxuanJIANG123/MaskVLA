# import packages and module here
import sys
sys.argv.append("--robot=aloha")
import os
import base64
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import cv2
from PIL import Image
from io import BytesIO

import cv2
import numpy as np
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MASKVLA_ROOT = Path(__file__).resolve().parents[3] / "MaskVLA"
sys.path.insert(0, str(MASKVLA_ROOT))

print("Python搜索路径(sys.path):")
for i, path in enumerate(sys.path):
    print(f"  [{i}] {path}")

frame_idx = 0
task_idx = 0
obs_idx = 0

import experiments.robot.openvla_utils
print(f"experiments.robot.openvla_utils模块路径: {experiments.robot.openvla_utils.__file__}")

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_processor,
    get_action_head,
    get_proprio_projector,
    # decode_base64_image,
)

from experiments.robot.robot_utils import get_image_resize_size

from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM

@dataclass
class OpenVLAConfig:

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/home/abrain/My_OFT/ckpt/hanging_mug/baseline-oft/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0807--50000_chkpt"
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

class OpenVLAPolicy:
    """MaskVLA 策略类"""
    
    def __init__(self, config: OpenVLAConfig):
        self.config = OpenVLAConfig
            # 模型组件加载
        self.vla = get_vla(self.config)
        self.processor = get_processor(self.config)
        self.action_head = get_action_head(self.config, self.vla.llm_dim) if self.config.use_l1_regression else None
        self.proprio_projector = get_proprio_projector(self.config, self.vla.llm_dim, PROPRIO_DIM) if self.config.use_proprio else None


        print(f"MaskVLA Policy loaded: {config.pretrained_checkpoint}")
        
    def update_obs(self, obs):
        """更新观察缓存"""
        self.obs_cache.append(obs)
        # 可以限制缓存长度
        if len(self.obs_cache) > 10:  
            self.obs_cache.pop(0)
    
    def get_action(self, obs=None, instruction=None):
        """获取动作预测"""
        global obs_idx,task_idx
        if obs is None and len(self.obs_cache) > 0:
            obs = self.obs_cache[-1]
            print("🍑🍑🍑 使用缓存的最新观察进行动作预测")
        
        elif obs is None:
            raise ValueError("No observation available")
        
        obs_idx += 1

        head_camera_b64 = obs["images"]["front_t"]
        left_camera_b64 = obs["images"]["left_t"]
        right_camera_b64 = obs["images"]["right_t"]
    
        
        # 原有的动作预测代码...
        try:
            # 推理动作
            actions = get_vla_action(
                self.config,
                self.vla,
                self.processor,
                obs=obs,
                task_label=instruction,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.config.use_film,
            )

            # 输出预测的actions
            print(f"🤖 模型预测动作形状: {[a.shape if hasattr(a, 'shape') else 'N/A' for a in actions]}")

            return actions
        except Exception as e:
            print(f"❌ 错误详情: {e}")
            print(f"❌ 错误类型: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e

        return actions


def encode_obs(observation):
    """Post-Process Observation for MaskVLA"""
    obs = observation

    global frame_idx,task_idx

    # 提取图像数据
    head_camera = obs["observation"]["head_camera"]["rgb"]
    left_camera = obs["observation"]["left_camera"]["rgb"] 
    right_camera = obs["observation"]["right_camera"]["rgb"]
    
    frame_idx += 1

    # 将图像编码为 base64（模拟客户端格式）
    def encode_image_to_base64(image):
        # 确保图像是 BGR 格式（OpenCV 默认）
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        _, buffer = cv2.imencode('.jpg', image_bgr)
        return base64.b64encode(buffer).decode('utf-8')
    
    qpos = obs["qpos"]  # 直接从观察中获取

    # 构建 OpenVLA 期望的观察格式
    processed_obs = {
        "instruction": "",  # 将在 eval 函数中设置
        "images": {
            "front_t": encode_image_to_base64(head_camera),   # 主摄像头
            "right_t": encode_image_to_base64(right_camera),  # 右摄像头  
            "left_t": encode_image_to_base64(left_camera),    # 左摄像头
        },
        "full_image": [head_camera, right_camera, left_camera],  # 直接用 numpy 数组
        "state": np.array(qpos),  # 添加这行
        "qpos": np.array(qpos)    # 同时保留这个，以防需要
    }    

    front_t = encode_image_to_base64(head_camera)
    right_t = encode_image_to_base64(right_camera)
    left_t = encode_image_to_base64(left_camera)

    save_dir = "/home/hjy/RoboTwin/debug_images/obs_img"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_base64_image_with_pil(base64_str, filepath):
        """使用PIL保存base64图片，可以指定格式"""
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        image.save(filepath, 'JPEG')  # 可以改为PNG等其他格式

    return processed_obs


def get_model(usr_args):
    """从 usr_args 加载 MaskVLA 模型"""
    
    # 从 usr_args 构建配置
    config = OpenVLAConfig()
    
    # 必需的配置参数
    config.pretrained_checkpoint = usr_args.get("pretrained_checkpoint", "")
    config.unnorm_key = usr_args.get("unnorm_key", "")
    
    # 可选配置参数
    config.model_family = usr_args.get("model_family", "openvla")
    config.use_proprio = usr_args.get("use_proprio", True)
    config.use_l1_regression = usr_args.get("use_l1_regression", True)
    config.use_diffusion = usr_args.get("use_diffusion", False)
    config.num_images_in_input = usr_args.get("num_images_in_input", 3)
    config.center_crop = usr_args.get("center_crop", True)
    config.use_film = usr_args.get("use_film", True)
    config.load_in_8bit = usr_args.get("load_in_8bit", False)
    config.load_in_4bit = usr_args.get("load_in_4bit", False)
    config.lora_rank = usr_args.get("lora_rank", 32)
    config.num_open_loop_steps = usr_args.get("num_open_loop_steps", 1)
    config.seed = usr_args.get("seed", 7)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 验证必需参数
    if not config.pretrained_checkpoint:
        raise ValueError("pretrained_checkpoint is required in usr_args")
    if not config.unnorm_key:
        raise ValueError("unnorm_key is required in usr_args")
    
    # 创建策略实例
    model = OpenVLAPolicy(config)
    
    # 添加调试信息
    print("=== Available norm_stats keys ===")
    if hasattr(model.vla, 'norm_stats'):
        print(list(model.vla.norm_stats.keys()))
    else:
        print("No norm_stats found in model")

    return model

def eval(TASK_ENV, model, observation):
    """执行策略评估"""

    # 后处理观察
    global frame_idx, task_idx

    task_idx = TASK_ENV.test_num
    obs = encode_obs(observation)
    
    # 获取任务指令
    instruction = TASK_ENV.get_instruction()
    obs["instruction"] = instruction
    
    print(f"🔍 任务指令: {instruction}")

    # 如果是第一帧，强制更新观察
    if len(model.obs_cache) == 0:
        model.update_obs(obs)
    
    # 获取动作预测
    actions = model.get_action(obs, instruction)
    
    # 执行每一步动作
    for action in actions:
        # eval.step_count += 1
        
        # 执行动作
        TASK_ENV.take_action(action, action_type='qpos')
        
        # 获取新观察
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        obs["instruction"] = instruction
        
        # 更新观察缓存
        model.update_obs(obs)

def reset_model(model):
    """重置模型缓存"""
    # 清理观察缓存
    model.obs_cache = []

    pass
