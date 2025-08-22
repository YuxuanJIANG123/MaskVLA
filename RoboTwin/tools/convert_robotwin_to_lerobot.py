#!/usr/bin/env python3
import h5py, json, pickle, pathlib, numpy as np, pandas as pd
import imageio.v2 as iio
from tqdm import tqdm

SRC_ROOT = pathlib.Path("/home/hjy/RoboTwin/data/beat_block_hammer/demo_randomized")  
DST_ROOT = pathlib.Path("/home/qyk/robotwin_dataset")                                    
FPS = 20  # 估个值；若有真实fps就用真实的

cams = ["front_camera", "left_camera", "right_camera", "head_camera"]

def bytes_to_img(b):
    return iio.imread(b.tobytes())  # b 是 numpy bytestring

def save_video(frames, out_path, fps=FPS):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(out_path, fps=fps)
    for im in frames:
        writer.append_data(im)
    writer.close()

def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "meta").mkdir(exist_ok=True)
    (DST_ROOT / "data").mkdir(exist_ok=True)
    (DST_ROOT / "videos").mkdir(exist_ok=True)

    episodes = sorted((SRC_ROOT / "data").glob("episode*.hdf5"))
    lengths = []
    meta_entries = []
    for epi_id, h5_path in enumerate(tqdm(episodes)):
        with h5py.File(h5_path, "r") as f:
            T = f["qpos"].shape[0]
            # --- states ---
            qpos = f["qpos"][()]   # (T,14)

            # --- decode RGB frames ---
            vid_paths = {}
            for cam in cams:
                rgb_bytes = f[f"observation/{cam}/rgb"][()]  # (T,) bytes
                frames = [bytes_to_img(b) for b in rgb_bytes]
                # save mp4
                outv = DST_ROOT / "videos" / f"traj_{epi_id:04d}_{cam}.mp4"
                save_video(frames, outv, FPS)
                H, W = frames[0].shape[:2]
                vid_paths[cam] = {"path": str(outv.relative_to(DST_ROOT)), "resolution": [H, W], "fps": FPS, "channels": 3}

            # --- actions (Δqpos) ---
            dq = np.zeros_like(qpos)
            dq[1:] = qpos[1:] - qpos[:-1]

            # dones / rewards （这里没有就自己造）
            dones = np.zeros((T,), dtype=bool)
            dones[-1] = True
            rewards = np.zeros((T,), dtype=float)

            # language instruction
            ins_file = SRC_ROOT / "instructions" / f"episode{epi_id}.json"
            if ins_file.exists():
                ins_json = json.load(open(ins_file))
                # 随便取一条 seen 的
                lang = ins_json["seen"][0] if ins_json.get("seen") else ""
            else:
                lang = ""

            # --- 写 parquet ---
            df = pd.DataFrame({
                "timestep": np.arange(T),
                "state.qpos": list(qpos),
                "action.dqpos": list(dq),
                "dones": dones,
                "rewards": rewards,
                "language_instruction": [lang]*T,
            })
            parquet_path = DST_ROOT / "data" / f"traj_{epi_id:04d}.parquet"
            df.to_parquet(parquet_path)

            lengths.append(T)
            meta_entries.append({
                "trajectory_id": epi_id,
                "length": T,
                "video": {f"video.{cam}": vid_paths[cam] for cam in cams},
                "state": {"state.qpos": {"shape": [T, qpos.shape[1]]}},
                "action": {"action.dqpos": {"shape": [T, dq.shape[1]]}},
                "language_instruction": lang
            })

    # --- meta/info.json ---
    info = {
        "total_episodes": len(episodes),
        "total_frames": int(np.sum(lengths)),
        "modalities": {
            "video": {f"video.{c}": {"fps": FPS,
                                     "resolution": meta_entries[0]["video"][f"video.{c}"]["resolution"],
                                     "channels": 3} for c in cams},
            "state": {"state.qpos": {"dim": 14}},
            "action": {"action.dqpos": {"dim": 14}},
            "language": {"language_instruction": {}}
        },
        "trajectories": meta_entries
    }
    with open(DST_ROOT / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- meta/modality.json(粗略来一个) ---
    modality = {
        "video": [f"video.{c}" for c in cams],
        "state": ["state.qpos"],
        "action": ["action.dqpos"],
        "language": ["language_instruction"]
    }
    with open(DST_ROOT / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

if __name__ == "__main__":
    main()
