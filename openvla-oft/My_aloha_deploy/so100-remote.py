from typing import Dict
import os
import time
import torch
import numpy as np
from PIL import Image
import pickle
import torchvision
import torchvision.transforms.functional as F

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from real_world_deployment import *

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from termcolor import cprint
import base64
from jsonargparse import ArgumentParser
from dataclasses import dataclass
import glob
from PIL import Image
from io import BytesIO
import cv2
from concurrent.futures import ThreadPoolExecutor
from turbojpeg import TurboJPEG


app = FastAPI()


@dataclass
class EvaluateConfig:
    # ckpt_path: str = (
    #     "/data/ckpt/univla/20250619+pick_candy_to_circle/so100+cam_left+b12+lr-0.0001+lora-r32+dropout-0.0=w-LowLevelDecoder-ws-25/22500"
    # )
    ckpt_path: str = (
        "/data/ckpt/univla/aloha+pick_put_banana_0604/aloha+cam2+b12+lr-0.0001+lora-r32+dropout-0.0=w-LowLevelDecoder-ws-25/27500"
    )
    decoder_path: str = "action_decoder-*.pt"

    camera_names: str = "cam2_t"
    window_size: int = 25
    task_description = "pick up the bananas and put them in the fruit basket"
    model_test: bool = False
    single_arm: bool = False

    def __post_init__(self):
        # 自动转换字符串为列表
        if isinstance(self.camera_names, str):
            self.camera_names = self.camera_names.split()


# 老代码，没用上，备份用
def decode_and_process_images(image_dict, target_size=(640, 480)):
    """
    解码 base64 编码的图像并预处理

    参数:
        image_dict: 包含 base64 编码图像的字典

    返回:
        处理后的图像字典 (numpy 数组格式)
    """
    processed_images = {}

    for cam_name, img_base64 in image_dict.items():
        try:
            # 1. 解码 base64 字符串
            img_data = base64.b64decode(img_base64)

            # 2. 转换为 numpy 数组 (使用 OpenCV)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # 3. 转换为 RGB 格式
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # 4. 调整尺寸
            img = cv.resize(img, target_size)

            # 5. 添加到结果字典
            processed_images[cam_name] = img

        except Exception as e:
            print(f"Error processing {cam_name}: {str(e)}")
            # 失败时使用零数组代替
            processed_images[cam_name] = np.zeros(
                (target_size[1], target_size[0], 3), dtype=np.uint8
            )

    return processed_images


def decode_one_image(app: FastAPI, cam_name, img_base64, target_size=(640, 480)):
    try:
        img_data = base64.b64decode(img_base64)

        if img_data[:2] == b"\xff\xd8":  # JPEG格式，用TurboJPEG解码
            jpeg = app.state.jpeg
            img = jpeg.decode(img_data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
        elif img_data[:4] == b"\x89PNG":
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img = img.resize(target_size)
        else:  # 其他格式默认用PIL打开
            img = Image.open(BytesIO(img_data)).convert("RGB")
            img = img.resize(target_size)

        img = np.asarray(img, dtype=np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error processing {cam_name}: {str(e)}")
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)


def decode_and_process_images_multi(app: FastAPI, image_dict, target_size=(640, 480)):
    # processed_images = {}

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     futures = {
    #         executor.submit(decode_one_image, app, cam_name, img_base64, target_size): cam_name
    #         for cam_name, img_base64 in image_dict.items()
    #     }

    #     for future in futures:
    #         cam_name = futures[future]
    #         processed_images[cam_name] = future.result()

    # return processed_images
    processed_images = {}
    opt: EvaluateConfig = app.state.opt

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                decode_one_image, app, cam_name, image_dict[cam_name], target_size
            ): cam_name
            for cam_name in opt.camera_names  # 只解码需要的图像
            if cam_name in image_dict
        }

        for future in futures:
            cam_name = futures[future]
            processed_images[cam_name] = future.result()

    return processed_images


def adjust_dict(app: FastAPI, data: Dict) -> Dict:
    batch = {}
    opt: EvaluateConfig = app.state.opt

    # 处理图像并归一化
    # all_processed_images = decode_and_process_images(data["images"])
    all_processed_images = decode_and_process_images_multi(app, data["images"])

    # 构建 numpy batch
    camera_np_batch = np.stack(
        [all_processed_images[camera_name] for camera_name in opt.camera_names],
        axis=0,
    )  # [camera_num, 480, 640, 3]

    camera_tensors = torch.from_numpy(camera_np_batch)  # [camera_num, 480, 640, 3]

    batch["observation.images"] = (
        camera_tensors.unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous()
    )  # [1, camera_num, 3, 480, 640]

    batch["observation.state"] = torch.from_numpy(
        np.array(data["qpos"], dtype=np.float32)
    ).unsqueeze(
        0
    )  # [1, 7]

    return batch


@app.post("/predict")
async def predict(request: Request):
    try:
        opt: EvaluateConfig = app.state.opt
        policy = app.state.policy
        stats = app.state.stats

        observation_data = await request.json()

        batch = adjust_dict(app, observation_data)

        # print(f"stats shape is {batch['observation.state'].shape}")  # [1, 7]
        # print(f"image shape is {batch['observation.images'].shape}")  # [1, 1, 3, 480, 640]
        state = batch["observation.state"].squeeze(0)  # [7]
        qpos_tensor = (state - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_tensor = qpos_tensor.unsqueeze(0).float()

        images = batch["observation.images"].squeeze(0).squeeze(0)  # [3, 480, 640]
        images_resized = F.resize(images, [224, 224])  # [3, 224, 224]

        task_description = opt.task_description

        actions = policy.step(images_resized, task_description, qpos_tensor)
        actions = actions * stats["action_std"] + stats["action_mean"]

        # print(f"state is {state}")
        # print(f"action is {actions}")

        # res_data = batch["observation.state"].view(-1).cpu().tolist()
        res_data = actions.tolist()

        return JSONResponse(content={"success": True, "actions": res_data})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.get("/health")  # 测试耗时 4-5ms
def health_check():
    return JSONResponse(content={"status": "ok"})


def load_model_and_stats(app: FastAPI, opt: EvaluateConfig):
    # 确保使用与保存时相同的配置
    stats_path = os.path.join(opt.ckpt_path, "dataset_stats.pkl")

    # 读取pickle文件
    with open(stats_path, "rb") as f:  # 注意使用二进制读取模式 'rb'
        stats = pickle.load(f)

    pattern = os.path.join(opt.ckpt_path, opt.decoder_path)

    # 匹配文件
    decoder_files = glob.glob(pattern)

    if len(decoder_files) == 0:
        raise FileNotFoundError(f"No decoder file matching {pattern}")
    elif len(decoder_files) > 1:
        print(
            f"⚠️ Found multiple decoder files: {decoder_files}, using {decoder_files[0]}"
        )

    # 取第一个
    decoder_path = decoder_files[0]

    cprint(f"Load action_decoder from {decoder_path}", "green")

    policy = UniVLAInference(
        saved_model_path=opt.ckpt_path,
        decoder_path=decoder_path,
        pred_action_horizon=opt.window_size,
        single_arm=opt.single_arm,
    )

    app.state.policy = policy
    app.state.stats = stats

    cprint("Model and stats loaded successfully", "green")


def model_test(app: FastAPI, opt: EvaluateConfig):

    load_model_and_stats(app, opt)
    policy = app.state.policy
    stats = app.state.stats

    # 图像 tensor：[1, num_cameras, 3, 480, 640]
    image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
    # qpos tensor：[1, 7]
    action_dim = 7 if opt.single_arm is True else 14
    qpos_tensor = torch.zeros((1, action_dim), dtype=torch.float32)

    cprint(f"qpos unnorm input is {qpos_tensor}", "yellow")
    qpos_tensor = (qpos_tensor - stats["qpos_mean"]) / stats["qpos_std"]

    task_description = "pick andy to circle"

    from tqdm import tqdm

    steps = 100
    with torch.inference_mode():
        for _ in tqdm(range(steps)):
            all_actions = policy.step(image_tensor, task_description, qpos_tensor)

        cprint(f"Time cost test:", "blue", "on_light_green")
        start_time = time.time()
        for _ in tqdm(range(steps)):
            all_actions = policy.step(image_tensor, task_description, qpos_tensor)
        cprint(
            f"One step cost time {(time.time() - start_time) * 1000 / steps}ms...",
            "blue",
            "on_light_green",
        )

    # print("policy out is", [f"{v:.3f}" for v in all_actions.tolist()])
    all_actions_0 = all_actions * stats["qpos_std"] + stats["qpos_mean"]
    cprint(
        f"actions norm with qpos is {[f'{v:.3f}' for v in all_actions_0.tolist()]}",
        "yellow",
    )
    all_actions_1 = all_actions * stats["action_std"] + stats["action_mean"]
    cprint(
        f"actions norm with action is {[f'{v:.3f}' for v in all_actions_1.tolist()]}",
        "yellow",
    )


def main() -> None:
    # 浮点数只打印三位
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    np.set_printoptions(linewidth=200)

    parser = ArgumentParser()
    parser.add_class_arguments(EvaluateConfig, as_group=False)  # 直接注册为顶级参数
    args = parser.parse_args()
    opt = EvaluateConfig(**vars(args))

    app.state.opt = opt
    app.state.policy = None
    app.state.stats = None
    app.state.jpeg = TurboJPEG(
        lib_path="/root/anaconda3/envs/ysh_robo/lib/libturbojpeg.so"
    )

    if opt.model_test:
        model_test(app, opt)
    else:
        load_model_and_stats(app, opt)
        cprint("Server starting...", "cyan")
        uvicorn.run(app, host="0.0.0.0", port=5000, reload=False)


if __name__ == "__main__":
    main()
