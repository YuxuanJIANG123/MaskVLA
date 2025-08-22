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

import imageio.v2 as imageio  

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 添加 OpenVLA-OFT 项目路径（请替换为实际路径）
sys.path.insert(0, "/home/Better-oft/openvla-oft")  # OpenVLA-OFT 项目根目录

print("Python搜索路径(sys.path):")
for i, path in enumerate(sys.path):
    print(f"  [{i}] {path}")

frame_idx = 0  # 当前帧编号

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
    pretrained_checkpoint: Union[str, Path] = "/new_data/ckpt/openvla-oft/sim/baseline-oft/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--472--50000_chkpt"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_proprio: bool = True
    use_film: bool = True
    num_images_in_input: int = 3
    center_crop: bool = True
    num_open_loop_steps: int = 25
    unnorm_key: str = "my_aloha_sim_adjust_bottle"
    use_relative_actions: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 42



class OpenVLAPolicy:
    """OpenVLA-OFT 策略类"""
    
    def __init__(self, config: OpenVLAConfig):

        self.config = OpenVLAConfig
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
        elif obs is None:
            raise ValueError("No observation available")
            
        # 添加调试代码
        print("🔍 开始调试...")
        print(f"🔍 Vision backbone 类型: {type(self.vla.vision_backbone)}")
 
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
            print(f"🤖 模型预测动作: {[a.tolist() if hasattr(a, 'tolist') else a for a in actions]}")
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
    """Post-Process Observation for OpenVLA-OFT"""
    obs = observation
    
    global frame_idx  # 使用全局变量来跟踪帧编号

    # 构建图像路径
    head_path = f"/new_data/test_image/cam1_{frame_idx:04d}.png"
    left_path = f"/new_data/test_image/cam2_{frame_idx:04d}.png"
    right_path = f"/new_data/test_image/cam3_{frame_idx:04d}.png"

    # 加载图像并赋值
    head_camera = obs["observation"]["head_camera"]["rgb"] = imageio.imread(head_path)
    left_camera = obs["observation"]["left_camera"]["rgb"] = imageio.imread(left_path)
    right_camera = obs["observation"]["right_camera"]["rgb"] = imageio.imread(right_path)

    # frame_idx += 1  # 更新帧编号

    # if frame_idx >= 170 :
    # #     frame_idx = 0

    # print(f'🥊🥊🥊🥊frame is {TS}')
    # # 提取图像数据
    # head_camera = obs["observation"]["head_camera"]["rgb"]
    # left_camera = obs["observation"]["left_camera"]["rgb"] 
    # right_camera = obs["observation"]["right_camera"]["rgb"]
    
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

    # 保存图片
    save_base64_image_with_pil(front_t, f"{save_dir}/front_{timestamp}.jpg")
    save_base64_image_with_pil(right_t, f"{save_dir}/right_{timestamp}.jpg")
    save_base64_image_with_pil(left_t, f"{save_dir}/left_{timestamp}.jpg")

    # # 添加这些调试信息
    # print(f"head_camera type: {type(head_camera)}")
    # print(f"head_camera shape: {head_camera.shape if hasattr(head_camera, 'shape') else 'no shape'}")
    # print(f"left_camera type: {type(left_camera)}")
    # print(f"right_camera type: {type(right_camera)}")
    
    # print(f"full_image type: {type(processed_obs['full_image'])}")
    # print(f"full_image[0] type: {type(processed_obs['full_image'][0])}")

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
    # 初始化步数计数器和视频录制器
    # if not hasattr(eval, 'step_count'):
    #     eval.step_count = 0
    # if not hasattr(eval, 'video_recorder'):
    #     eval.video_recorder = VideoRecorder()
    
    # 后处理观察
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
        
        # # 录制当前帧（图像 + action信息）
        # eval.video_recorder.record_frame(
        #     obs["full_image"], 
        #     action if hasattr(action, '__len__') else action.cpu().numpy(), 
        #     eval.step_count
        # )
        
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
    
    # # 关闭视频录制器
    # if hasattr(eval, 'video_recorder'):
    #     eval.video_recorder.close()
    #     delattr(eval, 'video_recorder')
    
    # # 重置步数计数器
    # if hasattr(eval, 'step_count'):
    #     delattr(eval, 'step_count')

    # 如果有其他需要重置的状态，在这里添加
    pass

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


if __name__ == "__main__":
    cfg = OpenVLAConfig
    # 模型组件加载
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim) if cfg.use_l1_regression else None
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM) if cfg.use_proprio else None
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

    #     # 在程序结束时关闭视频录制
    # if hasattr(eval, 'video_recorder'):
    #     eval.video_recorder.close()
