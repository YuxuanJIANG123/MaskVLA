import os
import sys
import traceback
from pathlib import Path
from typing import Union

from flask import Flask, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[1]
MASKVLA_ROOT = Path(
    os.environ.get("MASKVLA_MODEL_ROOT") or os.environ.get("MASKVLA_OPENVLA_ROOT", REPO_ROOT / "MaskVLA")
).expanduser().resolve()
DEFAULT_CKPT = os.environ.get("MASKVLA_DEFAULT_CKPT", "")
DEFAULT_UNNORM_KEY = os.environ.get("MASKVLA_DEFAULT_UNNORM_KEY", "")
PORT_NUM = int(os.environ.get("MASKVLA_SERVER_PORT", "8082"))

if not MASKVLA_ROOT.exists():
    raise FileNotFoundError(
        f"MaskVLA path does not exist: {MASKVLA_ROOT}. "
        "Set MASKVLA_MODEL_ROOT to the MaskVLA code directory."
    )

sys.path.insert(0, str(MASKVLA_ROOT))
if "--robot=aloha" not in sys.argv:
    sys.argv.append("--robot=aloha")

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

from prismatic.vla.constants import PROPRIO_DIM


class OpenVLAConfig:
    """Runtime config expected by MaskVLA helper functions."""

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = DEFAULT_CKPT
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_proprio: bool = True
    use_film: bool = True
    num_images_in_input: int = 3
    center_crop: bool = True
    num_open_loop_steps: int = 25
    unnorm_key: str = DEFAULT_UNNORM_KEY
    use_relative_actions: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 42


class OpenVLAPolicy:
    """MaskVLA 策略类"""

    def __init__(self, config: OpenVLAConfig):
        self.config = config
        self.obs_cache = []
        self.vla = get_vla(self.config)
        self.processor = get_processor(self.config)
        self.action_head = get_action_head(self.config, self.vla.llm_dim) if self.config.use_l1_regression else None
        self.proprio_projector = get_proprio_projector(self.config, self.vla.llm_dim, PROPRIO_DIM) if self.config.use_proprio else None
        print(f"MaskVLA Policy loaded: {config.pretrained_checkpoint}")

    def update_obs(self, obs):
        """更新观察缓存"""
        self.obs_cache.append(obs)
        if len(self.obs_cache) > 10:
            self.obs_cache.pop(0)

    def get_action(self, obs=None, instruction=None):
        """获取动作预测"""
        if obs is None and len(self.obs_cache) > 0:
            obs = self.obs_cache[-1]
            print("🍑🍑🍑 使用缓存的最新观察进行动作预测")
        elif obs is None:
            raise ValueError("No observation available")

        try:
            return get_vla_action(
                self.config,
                self.vla,
                self.processor,
                obs=obs,
                task_label=instruction,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.config.use_film,
            )
        except Exception as e:
            print(f"❌ 错误详情: {e}")
            print(f"❌ 错误类型: {type(e)}")
            traceback.print_exc()
            raise


app = Flask(__name__)

models = {}


def get_or_create_model(ckpt_path: str, unnorm_key: str) -> OpenVLAPolicy:
    """根据配置获取或创建模型"""
    if not ckpt_path:
        raise ValueError("ckpt_path is required. Pass it in the request or set MASKVLA_DEFAULT_CKPT.")
    if not unnorm_key:
        raise ValueError("unnorm_key is required. Pass it in the request or set MASKVLA_DEFAULT_UNNORM_KEY.")

    model_key = f"{ckpt_path}_{unnorm_key}"

    if model_key not in models:
        print(f"🔄 创建新模型: {ckpt_path}")

        config = OpenVLAConfig()
        config.pretrained_checkpoint = ckpt_path
        config.unnorm_key = unnorm_key

        model = OpenVLAPolicy(config)
        models[model_key] = model
        print(f"✅ 模型加载完成: {model_key}")

    return models[model_key]


def serialize_actions(actions):
    """Convert model outputs into JSON-compatible Python values."""
    serialized_actions = []
    for action in actions:
        if hasattr(action, "cpu"):
            serialized_actions.append(action.cpu().numpy().tolist())
        elif hasattr(action, "tolist"):
            serialized_actions.append(action.tolist())
        else:
            serialized_actions.append(action)
    return serialized_actions


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json(silent=True) or {}
        obs = data.get('observation')
        instruction = data.get('instruction')
        ckpt_path = data.get('ckpt_path') or DEFAULT_CKPT
        unnorm_key = data.get('unnorm_key') or DEFAULT_UNNORM_KEY

        model = get_or_create_model(ckpt_path, unnorm_key)
        actions = model.get_action(obs, instruction)

        return jsonify({
            "success": True,
            "actions": serialize_actions(actions)
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
        data = request.get_json(silent=True) or {}
        ckpt_path = data.get('ckpt_path')
        unnorm_key = data.get('unnorm_key')

        if ckpt_path and unnorm_key:
            model_key = f"{ckpt_path}_{unnorm_key}"
            if model_key in models:
                models[model_key].obs_cache = []
                return jsonify({"success": True, "message": f"模型 {model_key} 重置成功"})
            return jsonify({"success": True, "message": f"模型 {model_key} 未加载，无需重置"})

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
