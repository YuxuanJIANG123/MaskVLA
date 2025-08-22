from flask import Flask, request, jsonify
import numpy as np
import traceback
import sys
import os
import base64

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import cv2
from PIL import Image
from io import BytesIO
from datetime import datetime

sys.path.insert(0, "/home/Better-oft/openvla-oft")  # OpenVLA-OFT 项目根目录
sys.argv.append("--robot=aloha")
PORT_NUM = 8082

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

class OpenVLAConfig:

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/new_data/ckpt/openvla-oft/sim/hanging_mug/loss_siglip_dino_0.3_0.2/openvla-7b+my_aloha_sim_hanging_mug+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--0806_hanging_mug_loss_0.3_0.5--30000_chkpt"
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
    """OpenVLA-OFT 策略类"""
    
    def __init__(self, config: OpenVLAConfig):
        self.config = config
        # self.config = OpenVLAConfig
            # 模型组件加载
        self.vla = get_vla(self.config)
        self.processor = get_processor(self.config)
        self.action_head = get_action_head(self.config, self.vla.llm_dim) if self.config.use_l1_regression else None
        self.proprio_projector = get_proprio_projector(self.config, self.vla.llm_dim, PROPRIO_DIM) if self.config.use_proprio else None
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
            print("🍑🍑🍑 使用缓存的最新观察进行动作预测")
        
        elif obs is None:
            raise ValueError("No observation available")
        
        
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

            return actions
        except Exception as e:
            print(f"❌ 错误详情: {e}")
            print(f"❌ 错误类型: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e

        return actions

app = Flask(__name__)

# 全局模型字典，根据不同配置缓存模型
models = {}

def get_or_create_model(ckpt_path, unnorm_key):
    """根据配置获取或创建模型"""
    model_key = f"{ckpt_path}_{unnorm_key}"
    
    if model_key not in models:
        print(f"🔄 创建新模型: {ckpt_path}")
        
        config = OpenVLAConfig()
        config.pretrained_checkpoint = ckpt_path
        config.unnorm_key = unnorm_key
        
        model = OpenVLAPolicy(config)
        model.obs_cache = []
        
        models[model_key] = model
        print(f"✅ 模型加载完成: {model_key}")
    
    return models[model_key]

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.json
        obs = data.get('observation')
        instruction = data.get('instruction')
        ckpt_path = data.get('ckpt_path')
        unnorm_key = data.get('unnorm_key')
        
        # 获取对应配置的模型
        model = get_or_create_model(ckpt_path, unnorm_key)
        
        # 调用模型推理
        actions = model.get_action(obs, instruction)
        
        # 转换为可序列化的格式
        serialized_actions = []
        for action in actions:
            if hasattr(action, 'cpu'):
                serialized_actions.append(action.cpu().numpy().tolist())
            elif hasattr(action, 'tolist'):
                serialized_actions.append(action.tolist())
            else:
                serialized_actions.append(action)
        
        return jsonify({
            "success": True,
            "actions": serialized_actions
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/reset', methods=['POST'])
def reset():
    """重置模型"""
    try:
        data = request.json
        ckpt_path = data.get('ckpt_path')
        unnorm_key = data.get('unnorm_key')
        
        if ckpt_path and unnorm_key:
            model_key = f"{ckpt_path}_{unnorm_key}"
            if model_key in models:
                models[model_key].obs_cache = []
                return jsonify({"success": True, "message": f"模型 {model_key} 重置成功"})
        else:
            # 重置所有模型
            for model in models.values():
                model.obs_cache = []
            return jsonify({"success": True, "message": "所有模型重置成功"})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy", 
        "loaded_models": len(models),
        "model_keys": list(models.keys())
    })

if __name__ == '__main__':
    print("🚀 启动模型服务器...")
    app.run(host='0.0.0.0', port=PORT_NUM, debug=False)