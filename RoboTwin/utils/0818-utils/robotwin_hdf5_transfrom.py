import json
import os
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
import argparse
from glob import glob
import random
from tqdm import tqdm

def decode_image_from_bytes(image_bytes):
    """从字节数据解码图像 - 从sim_to_real_hdf5.py抄来的函数"""
    try:
        # 尝试直接从字节解码
        image = Image.open(BytesIO(image_bytes))
        # 转换为RGB格式
        image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"图像解码失败: {e}")
        # 返回None，让调用方处理
        return None

def decode_and_resize_images(image_bytes_array, size=256):
    """修改后的图像解码和调整大小函数"""
    resized = []
    for img_bytes in image_bytes_array:
        # 首先尝试原来的方法
        try:
            img_pil = Image.open(BytesIO(img_bytes.tobytes())).convert("RGB")
        except:
            # 如果失败，尝试新的解码方法
            try:
                # 如果img_bytes是字节数组，直接使用
                if isinstance(img_bytes, bytes):
                    img_pil = decode_image_from_bytes(img_bytes)
                else:
                    # 尝试转换为字节
                    img_pil = decode_image_from_bytes(img_bytes.tobytes())
                
                if img_pil is None:
                    # 如果解码失败，创建黑色图像
                    img_pil = Image.new('RGB', (size, size), (0, 0, 0))
            except Exception as e:
                print(f"图像处理失败，使用黑色图像: {e}")
                img_pil = Image.new('RGB', (size, size), (0, 0, 0))
        
        # 调整图像大小
        img_resized = img_pil.resize((size, size), resample=Image.BICUBIC)
        resized.append(np.array(img_resized))
    
    return np.stack(resized)

def load_instruction(instruction_dir, episode_idx):
    json_path = os.path.join(instruction_dir, f"episode_{episode_idx}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Instruction file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    candidates = data.get("seen", []) + data.get("unseen", [])
    if not candidates:
        raise ValueError(f"No instructions found in {json_path}")
    return random.choice(candidates)

def process_one_episode(input_path, output_path, episode_idx, base_data_dir, resize_size=256):
    # 从input_path推断任务名和instruction路径
    # input_path 格式: .../task_name/demo_clean/data/episode_xxx.hdf5
    task_name = input_path.split('/')[-4]  # 获取任务文件夹名
    
    # 构建instruction路径
    instruction_dir = os.path.join(base_data_dir, task_name, "demo_clean/instructions")
    
    # 从文件名提取episode编号
    episode_filename = os.path.basename(input_path)  # episode_xxx.hdf5
    episode_num = episode_filename.replace('episode_', '').replace('.hdf5', '')

    with h5py.File(input_path, "r") as f:
        action = f["qpos"][()]
        rel_action = np.zeros_like(action)
        rel_action[:-1] = action[1:] - action[:-1]
        rel_action[-1] = rel_action[-2]

        # 使用修改后的解码函数，按cam1,cam2,cam3方式映射
        cam1 = decode_and_resize_images(f["observation/head_camera/rgb"][()], size=resize_size)  # head_camera -> cam1
        cam2 = decode_and_resize_images(f["observation/left_camera/rgb"][()], size=resize_size)   # left_camera -> cam2
        cam3 = decode_and_resize_images(f["observation/right_camera/rgb"][()], size=resize_size)  # right_camera -> cam3

    # 读取 instruction JSON
    json_path = os.path.join(instruction_dir, f"episode{episode_idx}.json")
    with open(json_path, "r") as f:
        inst_data = json.load(f)
    seen_list = inst_data.get("seen", [])
    unseen_list = inst_data.get("unseen", [])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("cam1", data=cam1, dtype="uint8", chunks=(1, resize_size, resize_size, 3))
        f.create_dataset("cam2", data=cam2, dtype="uint8", chunks=(1, resize_size, resize_size, 3))
        f.create_dataset("cam3", data=cam3, dtype="uint8", chunks=(1, resize_size, resize_size, 3))
        f.create_dataset("action", data=action)
        f.create_dataset("relative_action", data=rel_action)
        f.create_dataset("seen", data=np.array(seen_list, dtype=h5py.string_dtype(encoding="utf-8")))
        f.create_dataset("unseen", data=np.array(unseen_list, dtype=h5py.string_dtype(encoding="utf-8")))


def main(args):

    file_dir = args.dataset_path  # 这个应该是 data_robotwin_clean/data
    
    # 遍历所有任务文件夹
    all_eps = []
    task_folders = [d for d in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, d))]
    
    for task_folder in task_folders:
        if task_folder in ['.cache']:  # 跳过非任务文件夹
            continue
        
        task_data_dir = os.path.join(file_dir, task_folder, "demo_clean/data")
        if os.path.exists(task_data_dir):
            task_hdf5_files = glob(os.path.join(task_data_dir, "*.hdf5"))
            all_eps.extend(task_hdf5_files)

    random.seed(42)
    random.shuffle(all_eps)

    n_val = int(len(all_eps) * args.percent_val)
    train_eps = all_eps[:-n_val]
    val_eps = all_eps[-n_val:]

    print(f"Total episodes: {len(all_eps)}")
    print(f"Train: {len(train_eps)}, Val: {len(val_eps)}")

    for split_name, split_eps in [("train", train_eps), ("val", val_eps)]:
        out_dir = os.path.join(output_base, split_name)
        os.makedirs(out_dir, exist_ok=True)
        for i, ep in enumerate(tqdm(split_eps, desc=f"Processing {split_name}")):
            ep_name = f"episode_{i}.hdf5"
            out_path = os.path.join(out_dir, ep_name)
            try:
                process_one_episode(ep, out_path, i, file_dir, resize_size=resize_size)
            except Exception as e:
                print(f"[ERROR] Failed to process {ep}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to RoboTwin hdf5 files")
    parser.add_argument("--out_base_dir", type=str, required=True,
                        help="Output dir for processed OpenVLA-compatible dataset")
    parser.add_argument("--percent_val", type=float, default=0.05,
                        help="Fraction of data to use as validation")
    parser.add_argument("--img_resize_size", type=int, default=256,
                        help="Final size for RGB images")
    args = parser.parse_args()
    main(args)


"""
python preprocess_aloha.py   --dataset_path /mnt/data/VLA_flowmatching/RoboTwin/data/place_object_scale/demo_randomized/data   --out_base_dir /mnt/data/VLA_flowmatching/RoboTwin/data/place_object_scale/processed_openvla/   --percent_val 0.05 --instruction_dir /mnt/data/VLA_flowmatching/RoboTwin/data/place_object_scale/demo_randomized/instructions
"""

