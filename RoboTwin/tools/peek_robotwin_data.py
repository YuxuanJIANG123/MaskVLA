import h5py, pickle, json, pathlib, numpy as np
from pprint import pprint

root = pathlib.Path("/home/hjy/RoboTwin/data/beat_block_hammer/demo_randomized")  # 改成你的路径

def peek_hdf5(p):
    print(f"\n=== HDF5: {p} ===")
    with h5py.File(p, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name:40s} shape={obj.shape} dtype={obj.dtype}")
        f.visititems(visit)

def peek_pkl(p):
    print(f"\n=== PKL: {p} ===")
    with open(p, "rb") as fp:
        obj = pickle.load(fp)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "shape"):
                print(f"{k:30s} shape={v.shape} dtype={getattr(v,'dtype',None)}")
            else:
                print(f"{k:30s} type={type(v)}")
    else:
        print("type:", type(obj))

def peek_json(p):
    print(f"\n=== JSON: {p} ===")
    with open(p, "r") as fp:
        js = json.load(fp)
    # 打印前几个键
    pprint({k: js[k] for k in list(js)[:10]})

# ---- run ----
# 任选几条 episode 看就行
for h5 in sorted(root.glob("data/episode*.hdf5"))[:2]:
    peek_hdf5(h5)

for pk in sorted(root.glob("_traj_data/episode*.pkl"))[:2]:
    peek_pkl(pk)

for jj in sorted(root.glob("instructions/episode*.json"))[:2]:
    peek_json(jj)

print("\n=== Check video info (first 1) ===")
# 如果你装了 ffprobe，可以打印分辨率/帧率：
import subprocess, shlex
mp4 = sorted((root/"video").glob("episode0/*.mp4"))[0]
cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate -of json "{mp4}"'
out = subprocess.check_output(shlex.split(cmd)).decode()
print(out)
