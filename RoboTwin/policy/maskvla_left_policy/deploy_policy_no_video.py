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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MASKVLA_ROOT = Path(__file__).resolve().parents[3] / "MaskVLA"
sys.path.insert(0, str(MASKVLA_ROOT))

print("Python搜索路径(sys.path):")
for i, path in enumerate(sys.path):
    print(f"  [{i}] {path}")


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
    """MaskVLA 配置类 - 映射到 deploy.py 中的 DeployConfig"""
    # 基础配置
    # model_family: str = "openvla"
    # pretrained_checkpoint: Union[str, Path] = ""
    
    # # 动作头配置
    # use_l1_regression: bool = True
    # use_diffusion: bool = False
    # num_diffusion_steps_train: int = 50
    # num_diffusion_steps_inference: int = 50
    
    # # 输入配置
    # num_images_in_input: int = 3
    # use_proprio: bool = True
    # center_crop: bool = True
    
    # # LoRA 配置
    # lora_rank: int = 32
    
    # # 归一化配置
    # unnorm_key: str = ""
    # use_relative_actions: bool = False
    
    # # 量化配置
    # load_in_8bit: bool = False
    # load_in_4bit: bool = False
    
    # # FiLM 配置
    # use_film: bool = True
    
    # # 执行配置
    # num_open_loop_steps: int = 1
    
    # # 种子
    # seed: int = 7
    
    # # RobotWin 平台特定配置
    # host: str = "0.0.0.0"
    # port: int = 8777

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/new_data/ckpt/MaskVLA/sim/openvla-7b+my_aloha_sim_adjust_bottle_and_stapler_pad+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--473--50000_chkpt"
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
    """MaskVLA 策略类"""
    
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
        
        print(f"MaskVLA Policy loaded: {config.pretrained_checkpoint}")
        
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
        
        # 使用正确的数据类型
        device = next(self.vla.parameters()).device
        dtype = next(self.vla.parameters()).dtype  # 获取模型的数据类型
        
        test_input = torch.randn(1, 18, 224, 224, device=device, dtype=dtype)  # 使用模型的dtype
        dummy_language_emb = torch.randn(1, 10, 4096, device=device, dtype=dtype)  # 同样使用模型的dtype
        
        with torch.no_grad():
            vision_output = self.vla.vision_backbone(test_input, dummy_language_emb)
            print(f"🔍 vision_backbone 输出类型: {type(vision_output)}")
            if isinstance(vision_output, tuple):
                print(f"🔍 vision_backbone 输出元组长度: {len(vision_output)}")
                for i, item in enumerate(vision_output):
                    if hasattr(item, 'shape'):
                        print(f"🔍   第{i}个元素: 类型={type(item)}, 形状={item.shape}")
                    else:
                        print(f"🔍   第{i}个元素: {item}")
            else:
                print(f"🔍 vision_backbone 输出张量形状: {vision_output.shape}")
        
        # 测试 _process_vision_features
        with torch.no_grad():
            process_output = self.vla._process_vision_features(test_input, dummy_language_emb, self.config.use_film)
            print(f"🔍 _process_vision_features 输出类型: {type(process_output)}")
            if isinstance(process_output, tuple):
                print(f"🔍 _process_vision_features 输出元组长度: {len(process_output)}")
            else:
                print(f"🔍 _process_vision_features 输出张量形状: {process_output.shape}")
    
        
        # 使用 OpenVLA 的动作预测函数
        # actions = get_vla_action(
        #     cfg=self.config,
        #     vla=self.vla,
        #     processor=self.processor,
        #     obs=obs,
        #     task_label=instruction,
        #     action_head=self.action_head,
        #     proprio_projector=self.proprio_projector,
        #     use_film=self.config.use_film,
        # )

        # 原有的动作预测代码...
        try:
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
                    # 输出预测的actions
            print(f"🤖 模型预测动作: {[a.tolist() if hasattr(a, 'tolist') else a for a in actions]}")
            
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
    """
    执行策略评估
    
    Args:
        TASK_ENV: 任务环境类
        model: 从 get_model() 返回的模型
        observation: 环境观察
    """

    # 添加步数计数器
    if not hasattr(eval, 'step_count'):
        eval.step_count = 0
    
    # 创建保存目录
    save_dir = "/test_output"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/images", exist_ok=True)

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
    # print(f"Vision backbone: {model.vla.vision_backbone}")

    # 获取动作预测
    actions = model.get_action(obs, instruction)
    
    # 执行每一步动作
    for action in actions:
        eval.step_count += 1
        
        # 每50步保存一次图像
        if eval.step_count % 50 == 0:
            # 保存当前观察的图像
            head_img = obs["full_image"][0]  # head_camera
            cv2.imwrite(f"{save_dir}/images/step_{eval.step_count:04d}_head.jpg", 
                    cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR))
            print(f"💾 已保存第 {eval.step_count} 步的图像")
        
        # 输出action值到终端
        print(f"Step {eval.step_count}: Action = {action}")


        # 根据 MaskVLA 的输出类型选择合适的动作类型
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
