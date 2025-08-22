#!/usr/bin/env python3
import json, pathlib, pandas as pd, numpy as np
from tqdm import tqdm
import subprocess, shlex

ROOT = pathlib.Path("/home/qyk/robotwin_dataset")   # 改成你的
DATA_DIR = ROOT / "data"
VIDEO_DIR = ROOT / "videos"
META_DIR = ROOT / "meta"
META_DIR.mkdir(exist_ok=True)

# 你的视频摄像头名字 & 想暴露给模型的 key
CAM_MAP = {
    "front_camera": "video.front",
    "left_camera":  "video.left",
    "right_camera": "video.right",
    "head_camera":  "video.head",
}
PARQUET_GLOB = "traj_*.parquet"
FPS_DEFAULT = 20  # 没有就先用默认

def ffprobe_fps(path):
    """从 mp4 取 fps；取不到就返回默认"""
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "{path}"'
        out = subprocess.check_output(shlex.split(cmd)).decode().strip()
        num, den = out.split("/")
        return float(num) / float(den)
    except Exception:
        return FPS_DEFAULT

def main():
    parquets = sorted(DATA_DIR.glob(PARQUET_GLOB))
    assert len(parquets) > 0, "No parquet files found!"

    episodes_lines = []
    ep_stats = []
    lengths = []
    task_id = 0  # 单任务
    total_frames = 0

    # 读取一次，拿到 feature 维度
    sample_df = pd.read_parquet(parquets[0])
    state_dim = len(sample_df.iloc[0]["state.qpos"])
    action_dim = len(sample_df.iloc[0]["action.dqpos"])

    # 采集视频参数（分辨率等）——随便取第一集的一个视频
    any_vid = next(VIDEO_DIR.glob("traj_0000_*_camera.mp4"))
    # 读分辨率
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json "{any_vid}"'
        out = subprocess.check_output(shlex.split(cmd)).decode()
        j = json.loads(out)["streams"][0]
        resolution = [j["height"], j["width"]]
    except Exception:
        resolution = [sample_df.iloc[0]["video_height"], sample_df.iloc[0]["video_width"]] if "video_height" in sample_df else [256,256]

    # 遍历每个 episode
    for p in tqdm(parquets, desc="scan episodes"):
        eid = int(p.stem.split("_")[1])
        df = pd.read_parquet(p)
        T = len(df)
        lengths.append(T)
        total_frames += T

        videos = {}
        fps_record = []
        for raw_name, key in CAM_MAP.items():
            v = VIDEO_DIR / f"traj_{eid:04d}_{raw_name}.mp4"
            if v.exists():
                videos[key] = str(v.relative_to(ROOT))
                fps_record.append(ffprobe_fps(str(v)))
            else:
                print(f"[WARN] missing video: {v}")
        fps_ep = np.mean(fps_record) if fps_record else FPS_DEFAULT

        episodes_lines.append({
            "episode_id": eid,
            "task_id": task_id,
            "length": T,
            "parquet": str(p.relative_to(ROOT)),
            "videos": videos
        })
        ep_stats.append({"episode_id": eid, "length": T})

    # ----- 写 episodes.jsonl
    with open(META_DIR/"episodes.jsonl", "w") as f:
        for line in episodes_lines:
            f.write(json.dumps(line)+"\n")

    # ----- 写 tasks.jsonl
    with open(META_DIR/"tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_id": task_id, "task_name": "beat_block_hammer"})+"\n")

    # ----- 写 stats.json（只做 state/action）
    def stat_dict(arrs):
        arr = np.concatenate(arrs, axis=0)
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    # 收集所有 state / action
    all_state = []
    all_action = []
    for p in tqdm(parquets, desc="collect stats"):
        df = pd.read_parquet(p)
        all_state.append(np.stack(df["state.qpos"].to_numpy()))
        all_action.append(np.stack(df["action.dqpos"].to_numpy()))

    stats = {
        "observation.state": stat_dict(all_state),
        "action": stat_dict(all_action)
    }
    json.dump(stats, open(META_DIR/"stats.json", "w"), indent=2)

    # ----- episodes_stats.json
    json.dump(ep_stats, open(META_DIR/"episodes_stats.json", "w"), indent=2)

    # ----- modality.json
    modality = {
        "state": {
            "dual_arm": {"start": 0, "end": state_dim}  # 你想细分再拆
        },
        "action": {
            "dual_arm": {"start": 0, "end": action_dim}
        },
        "video": {  # 映射到原始 key
            "front": {"original_key": "video.front"},
            "left":  {"original_key": "video.left"},
            "right": {"original_key": "video.right"},
            "head":  {"original_key": "video.head"}
        },
        "annotation": {
            "language_instruction": {"original_key": "language_instruction"}
        }
    }
    json.dump(modality, open(META_DIR/"modality.json", "w"), indent=2)

    # ----- info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "dual_arm",
        "total_episodes": len(parquets),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(parquets)*len(CAM_MAP),
        "fps": fps_ep,
        "splits": {"train": f"0:{len(parquets)}"},
        # 模板路径
        "data_path": "data/traj_{episode_id:04d}.parquet",
        "video_path": "videos/traj_{episode_id:04d}_{camera_raw}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": [f"state_{i}" for i in range(state_dim)]
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": [f"action_{i}" for i in range(action_dim)]
            },
            "language_instruction": {
                "dtype": "str",
                "shape": [1],
                "names": None
            },
            "frame_index": {"dtype":"int64","shape":[1],"names":None},
            "episode_index": {"dtype":"int64","shape":[1],"names":None},
            "task_index": {"dtype":"int64","shape":[1],"names":None}
        }
    }
    # 视频 feature 信息
    for raw, key in CAM_MAP.items():
        info["features"][f"{key.replace('video.','observation.images.') }"] = {
            "dtype": "video",
            "shape": [resolution[0], resolution[1], 3],
            "names": ["height","width","channel"],
            "info": {
                "video.height": resolution[0],
                "video.width": resolution[1],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps_ep,
                "video.channels": 3,
                "has_audio": False
            }
        }
    json.dump(info, open(META_DIR/"info.json", "w"), indent=2)

    print("All meta files written to", META_DIR)

if __name__ == "__main__":
    main()
