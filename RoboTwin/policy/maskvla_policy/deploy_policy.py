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
import requests

PORT_ID = 8081  # 远程服务器端口

@dataclass
class OpenVLAConfig:

    # model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = None
    # use_l1_regression: bool = True
    # use_diffusion: bool = False
    # use_proprio: bool = True
    # use_film: bool = True
    # num_images_in_input: int = 3
    # center_crop: bool = True
    # num_open_loop_steps: int = 25
    unnorm_key: str = None
    # use_relative_actions: bool = False
    # load_in_8bit: bool = False
    # load_in_4bit: bool = False
    # seed: int = 42


def send_and_receive(obs, instruction, ckpt_path, unnorm_key, server_url=f"http://192.168.3.101:{PORT_ID}"):
    """发送观察数据到远程服务器，接收动作序列"""
    
    # 转换numpy数组为列表
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    # 转换观察数据
    obs_serializable = convert_numpy_to_list(obs)
    
    data = {
        "observation": obs_serializable, 
        "instruction": instruction, 
        "ckpt_path": ckpt_path, 
        "unnorm_key": unnorm_key
    }
    
    response = requests.post(f"{server_url}/predict", json=data, timeout=600)
    result = response.json()
    
    if not result["success"]:
        raise Exception(f"推理失败: {result['error']}")
    
    return [np.array(action) for action in result["actions"]]

def reset_remote_model(ckpt_path, unnorm_key, server_url=f"http://192.168.3.101:{PORT_ID}"):
    """重置远程模型"""
    data = {
        "ckpt_path": ckpt_path, 
        "unnorm_key": unnorm_key
    }
    # response = requests.post(f"{server_url}/reset", json=data, timeout=10)
    # result = response.json()
    
    # if not result["success"]:
    #     raise Exception(f"重置失败: {result['error']}")
    
    # print(f"✅ {result['message']}")

def encode_obs(observation):
    """Post-Process Observation for MaskVLA"""
    obs = observation

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

    front_t = encode_image_to_base64(head_camera)
    right_t = encode_image_to_base64(right_camera)
    left_t = encode_image_to_base64(left_camera)

    return processed_obs

def get_config(usr_args):
    """从 usr_args 加载 MaskVLA 模型"""
    
    # 从 usr_args 构建配置
    config = OpenVLAConfig()
    
    # 必需的配置参数
    config.pretrained_checkpoint = usr_args.get("pretrained_checkpoint", "")
    config.unnorm_key = usr_args.get("unnorm_key", "")

    return config


def eval_remote(TASK_ENV, ckpt_path, unnorm_key, observation):
    """执行策略评估"""
    

    obs = encode_obs(observation)
    
    # 获取任务指令
    # hjy changes
    instruction = TASK_ENV.get_instruction()
    # instruction = "Pick up the bottle on the table headup with the correct arm."
    obs["instruction"] = instruction
    print(f"🔍 任务指令: {instruction}")

    # 获取动作预测
    actions = send_and_receive(obs, instruction, ckpt_path, unnorm_key, server_url=f"http://192.168.3.101:{PORT_ID}")
    
    # 执行每一步动作
    for action in actions:
        
        # 执行动作
        TASK_ENV.take_action(action, action_type='qpos')
        
        # 获取新观察
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        obs["instruction"] = instruction

        

def reset_model(model):
    """重置模型缓存"""
    # 清理观察缓存
    model.obs_cache = []
    # 如果有其他需要重置的状态，在这里添加
    pass
