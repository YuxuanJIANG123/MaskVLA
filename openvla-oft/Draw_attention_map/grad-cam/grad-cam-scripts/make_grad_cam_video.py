import os
import cv2
import numpy as np
import torch
import sys
import argparse
from tqdm import tqdm

sys.path.append('/home/Better-oft/openvla-oft/Draw_attention_map/attn-heat-map/dinov2')

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from models.vision_transformer import vit_large

# 配置
image_size = 224
d_height = d_width = image_size // 14

video_source = "/new_data/data_robotwin/stack_bowls_three/demo_randomized/video/episode0"

video_paths = {
    'head_cam': f'{video_source}/head_cam.mp4',
    'left_cam': f'{video_source}/left_cam.mp4',
    'right_cam': f'{video_source}/right_cam.mp4'
}

ckpt_name = ""

output_dir = '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/set0'
model_path = "/home/Better-oft/openvla-oft/openvla_extracted_dinov2_weights/robotwin_set_zero/full_state_dict.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_dinov2_weights(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    if 'pos_embed' in state_dict and state_dict['pos_embed'].shape != model.pos_embed.shape:
        pos_embed = state_dict['pos_embed']
        class_pos_embed = model.pos_embed[:, 0:1, :]
        new_pos_embed = torch.cat([class_pos_embed, pos_embed], dim=1)
        state_dict['pos_embed'] = new_pos_embed
    model.load_state_dict(state_dict, strict=False)
    return model

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame_rgb).unsqueeze(0).to(device)

def reshape_transform(tensor, height=d_height, width=d_width):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, height, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)

def extract_frames(video_path, fps=25):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / fps))
    
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames

def process_video(video_name, video_path, model, cam):
    print(f"处理 {video_name}...")
    
    parser = argparse.ArgumentParser(description="Grad-CAM 视频处理脚本")
    parser.add_argument('--video_path', type=str, default=video_path, help='视频文件路径')

    args = parser.parse_args()  

    # 提取帧
    frames = extract_frames(video_path)
    if not frames:
        print(f"无法提取帧: {video_path}")
        return
    
    # 处理每一帧
    cam_frames = []
    for frame in tqdm(frames, desc=f"处理{video_name}"):
        # 预处理
        input_tensor = preprocess_frame(frame)
        rgb_img = np.float32(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (image_size, image_size))) / 255
        
        # 生成CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_frames.append(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    # 保存视频
    os.makedirs(os.path.join(output_dir, video_name), exist_ok=True)
    output_path = os.path.join(output_dir, video_name, 'gradcam.mp4')
    
    height, width = cam_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))
    
    for frame in cam_frames:
        out.write(frame)
    out.release()
    
    print(f"保存至: {output_path}")

def main():
    # 加载模型
    print("加载模型...")
    model = vit_large(patch_size=14, img_size=224, init_values=1.0, block_chunks=0)
    model = load_dinov2_weights(model, model_path)
    model.to(device).eval()
    
    # 创建GradCAM
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    # 处理所有视频
    os.makedirs(output_dir, exist_ok=True)
    for video_name, video_path in video_paths.items():
        if os.path.exists(video_path):
            process_video(video_name, video_path, model, cam)
        else:
            print(f"视频不存在: {video_path}")
    
    print("完成!")

if __name__ == '__main__':
    main()