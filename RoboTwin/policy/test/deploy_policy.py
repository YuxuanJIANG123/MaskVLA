# import packages and module here
import sys
import os
import base64
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import cv2
from PIL import Image

# 添加 OpenVLA-OFT 项目路径（请替换为实际路径）
sys.path.insert(0, "/home/hjy/RoboTwin/policy/openvla_oft_policy")  # OpenVLA-OFT 项目根目录

# 导入 OpenVLA 相关模块 - 请根据实际项目结构调整路径
# sys.path.append("path/to/openvla-oft")  # 添加 OpenVLA-OFT 项目路径
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_processor,
    get_action_head,
    get_proprio_projector,
    # decode_base64_image,
)
from experiments.robot.robot_utils import get_image_resize_size


@dataclass
class OpenVLAConfig:
    """OpenVLA-OFT 配置类 - 映射到 deploy.py 中的 DeployConfig"""
    # 基础配置
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    
    # 动作头配置
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    
    # 输入配置
    num_images_in_input: int = 3
    use_proprio: bool = True
    center_crop: bool = True
    
    # LoRA 配置
    lora_rank: int = 32
    
    # 归一化配置
    unnorm_key: str = ""
    use_relative_actions: bool = False
    
    # 量化配置
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # FiLM 配置
    use_film: bool = False
    
    # 执行配置
    num_open_loop_steps: int = 1
    
    # 种子
    seed: int = 7
    
    # RobotWin 平台特定配置
    host: str = "0.0.0.0"
    port: int = 8777


class OpenVLAPolicy:
    """OpenVLA-OFT 策略类"""
    
    def __init__(self, config: OpenVLAConfig):
        self.config = config
        self.obs_cache = []  # 观察缓存，如果需要历史信息
        
        # 加载模型组件
        self.vla = get_vla(config)
        self.processor = get_processor(config)
        
        # 加载可选组件
        self.action_head = None
        if config.use_l1_regression or config.use_diffusion:
            self.action_head = get_action_head(config, self.vla.llm_dim)
            
        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = get_proprio_projector(
                config, self.vla.llm_dim, 
                # 这里需要根据实际机器人确定 PROPRIO_DIM
                14  # ALOHA 双臂机器人的状态维度
            )
        
        print(f"OpenVLA-OFT Policy loaded: {config.pretrained_checkpoint}")
        
    def update_obs(self, obs):
        """更新观察缓存"""
        self.obs_cache.append(obs)
        # 可以限制缓存长度
        if len(self.obs_cache) > 10:  
            self.obs_cache.pop(0)
    
    def get_action(self, obs=None, instruction=None):
        """获取动作预测"""
        if obs is None and len(self.obs_cache) > 0:
            obs = self.obs_cache[-1]
        elif obs is None:
            raise ValueError("No observation available")
            
        # 使用 OpenVLA 的动作预测函数
        actions = get_vla_action(
            cfg=self.config,
            vla=self.vla,
            processor=self.processor,
            obs=obs,
            task_label=instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.config.use_film,
        )
        
        return actions


def encode_obs(observation):
    """Post-Process Observation for OpenVLA-OFT"""
    obs = observation
    
    # # 调试：打印观察结构
    # print("=== observation keys ===")
    # print(list(obs.keys()))
    # if "joint_action" in obs:
    #     print("=== joint_action keys ===")
    #     print(list(obs["joint_action"].keys()))
    # else:
    #     print("No 'joint_action' key found")
    #     # 查找其他可能的关节状态字段
    #     for key in obs.keys():
    #         if "joint" in key.lower() or "qpos" in key.lower() or "state" in key.lower():
    #             print(f"Found potential joint field: {key}")

    # 提取图像数据
    head_camera = obs["observation"]["head_camera"]["rgb"]
    left_camera = obs["observation"]["left_camera"]["rgb"] 
    right_camera = obs["observation"]["right_camera"]["rgb"]
    
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

    # 添加这些调试信息
    print(f"head_camera type: {type(head_camera)}")
    print(f"head_camera shape: {head_camera.shape if hasattr(head_camera, 'shape') else 'no shape'}")
    print(f"left_camera type: {type(left_camera)}")
    print(f"right_camera type: {type(right_camera)}")
    
    print(f"full_image type: {type(processed_obs['full_image'])}")
    print(f"full_image[0] type: {type(processed_obs['full_image'][0])}")

    return processed_obs


def get_model(usr_args):
    """从 usr_args 加载 OpenVLA-OFT 模型"""
    
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
    config.use_film = usr_args.get("use_film", False)
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
    """
    执行策略评估
    
    Args:
        TASK_ENV: 任务环境类
        model: 从 get_model() 返回的模型
        observation: 环境观察
    """
    # 后处理观察
    obs = encode_obs(observation)
    
    # 获取任务指令
    instruction = TASK_ENV.get_instruction()
    obs["instruction"] = instruction
    
    # 如果是第一帧，强制更新观察
    if len(model.obs_cache) == 0:
        model.update_obs(obs)
    
    # 添加调试信息
    print("=== Available norm_stats keys ===")
    if hasattr(model.vla, 'norm_stats'):
        print(list(model.vla.norm_stats.keys()))
    else:
        print("No norm_stats found in model")

    print(f"Vision backbone type: {type(model.vla.vision_backbone)}")
    print(f"Vision backbone: {model.vla.vision_backbone}")

    # 获取动作预测
    actions = model.get_action(obs, instruction)
    
    # 执行每一步动作
    for action in actions:
        # 根据 OpenVLA-OFT 的输出类型选择合适的动作类型
        # 如果输出是关节角度，使用 'qpos'
        # 如果输出是末端位姿，使用 'ee'
        TASK_ENV.take_action(action, action_type='qpos')  # 假设是关节控制
        
        # 获取新的观察
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        obs["instruction"] = instruction
        
        # 更新观察缓存
        model.update_obs(obs)


def reset_model(model):
    """重置模型缓存"""
    # 清理观察缓存
    model.obs_cache = []
    
    # 如果有其他需要重置的状态，在这里添加
    pass