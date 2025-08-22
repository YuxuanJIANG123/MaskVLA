import cv2
import numpy as np
import os
from tqdm import tqdm

def create_video_grid():
    # 视频路径配置
    original_videos = {
        'head_cam': '/new_data/data_robotwin/stack_bowls_three/demo_randomized/video/episode0/head_cam.mp4',
        'left_cam': '/new_data/data_robotwin/stack_bowls_three/demo_randomized/video/episode0/left_cam.mp4',
        'right_cam': '/new_data/data_robotwin/stack_bowls_three/demo_randomized/video/episode0/right_cam.mp4'
    }
    
    baseline_videos = {
        'head_cam': '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/baseline/head_cam/gradcam.mp4',
        'left_cam': '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/baseline/left_cam/gradcam.mp4',
        'right_cam': '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/baseline/right_cam/gradcam.mp4'
    }
    
    set0_videos = {
        'head_cam': '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/set0/head_cam/gradcam.mp4',
        'left_cam': '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/set0/left_cam/gradcam.mp4',
        'right_cam': '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/set0/right_cam/gradcam.mp4'
    }
    
    output_path = '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/comparison_grid.mp4'
    
    # 检查所有视频是否存在
    all_videos = [original_videos, baseline_videos, set0_videos]
    video_names = ['head_cam', 'left_cam', 'right_cam']
    
    for video_dict in all_videos:
        for name in video_names:
            if not os.path.exists(video_dict[name]):
                print(f"警告: 视频文件不存在: {video_dict[name]}")
                return
    
    # 打开所有视频
    caps = []
    for video_dict in all_videos:
        caps_row = []
        for name in video_names:
            cap = cv2.VideoCapture(video_dict[name])
            if not cap.isOpened():
                print(f"无法打开视频: {video_dict[name]}")
                return
            caps_row.append(cap)
        caps.append(caps_row)
    
    # 获取视频信息
    first_cap = caps[0][0]
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"FPS: {fps}")
    print(f"总帧数: {total_frames}")
    
    # 设置统一的帧尺寸
    frame_width = 320
    frame_height = 240
    
    # 计算输出视频尺寸 (3x3 网格)
    output_width = frame_width * 3
    output_height = frame_height * 3
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # 添加文字标签的函数
    def add_label(frame, text, position='top'):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # 获取文字尺寸
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        if position == 'top':
            x = (w - text_width) // 2
            y = text_height + 10
        else:  # bottom
            x = (w - text_width) // 2
            y = h - 10
        
        # 添加黑色背景
        cv2.rectangle(frame, (x-5, y-text_height-5), (x+text_width+5, y+5), (0, 0, 0), -1)
        # 添加白色文字
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        
        return frame
    
    print("开始处理视频...")
    
    # 处理每一帧
    for frame_idx in tqdm(range(total_frames)):
        # 读取所有视频的当前帧
        grid_frames = []
        
        for row_idx, caps_row in enumerate(caps):
            frame_row = []
            for col_idx, cap in enumerate(caps_row):
                ret, frame = cap.read()
                if not ret:
                    # 如果某个视频结束了，创建黑色帧
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                else:
                    # 调整帧尺寸
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                # 添加标签
                cam_names = ['Head Cam', 'Left Cam', 'Right Cam']
                model_names = ['Original', 'Baseline', 'Set0']
                
                label = f"{model_names[row_idx]} - {cam_names[col_idx]}"
                frame = add_label(frame, label, 'top')
                
                frame_row.append(frame)
            grid_frames.append(frame_row)
        
        # 拼接帧
        # 先水平拼接每一行
        rows = []
        for frame_row in grid_frames:
            row = np.hstack(frame_row)
            rows.append(row)
        
        # 再垂直拼接所有行
        grid_frame = np.vstack(rows)
        
        # 写入输出视频
        out.write(grid_frame)
    
    # 释放资源
    for caps_row in caps:
        for cap in caps_row:
            cap.release()
    out.release()
    
    print(f"对比视频已保存至: {output_path}")

def create_simple_grid():
    """简化版本 - 如果上面的版本有问题可以使用这个"""
    # 视频路径
    videos = [
        # 第一行 - 原始视频
        '/new_data/data_robotwin/stack_bowls_three/demo_randomized/video/episode0/head_cam.mp4',
        '/new_data/data_robotwin/stack_bowls_three/demo_randomized/video/episode0/left_cam.mp4',
        '/new_data/data_robotwin/stack_bowls_three/demo_randomized/video/episode0/right_cam.mp4',
        # 第二行 - baseline
        '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/baseline/head_cam/gradcam.mp4',
        '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/baseline/left_cam/gradcam.mp4',
        '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/baseline/right_cam/gradcam.mp4',
        # 第三行 - set0
        '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/set0/head_cam/gradcam.mp4',
        '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/set0/left_cam/gradcam.mp4',
        '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/set0/right_cam/gradcam.mp4'
    ]
    
    labels = [
        'Original Head', 'Original Left', 'Original Right',
        'Baseline Head', 'Baseline Left', 'Baseline Right',
        'Set0 Head', 'Set0 Left', 'Set0 Right'
    ]
    
    output_path = '/home/Better-oft/openvla-oft/Draw_attention_map/grad-cam/grad-cam-video-results/simple_comparison_grid.mp4'
    
    # 打开所有视频
    caps = [cv2.VideoCapture(video) for video in videos]
    
    # 检查视频是否正常打开
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"无法打开视频: {videos[i]}")
            return
    
    # 获取视频参数
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
    
    # 设置每个子视频的尺寸
    sub_width, sub_height = 320, 240
    grid_width, grid_height = sub_width * 3, sub_height * 3
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
    
    print(f"处理 {total_frames} 帧...")
    
    for frame_idx in tqdm(range(total_frames)):
        frames = []
        
        # 读取所有视频的当前帧
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (sub_width, sub_height))
                frames.append(frame)
            else:
                # 如果读取失败，创建黑色帧
                frames.append(np.zeros((sub_height, sub_width, 3), dtype=np.uint8))
        
        # 添加标签
        for i, frame in enumerate(frames):
            cv2.putText(frame, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 排列成3x3网格
        rows = []
        for i in range(3):
            row = np.hstack(frames[i*3:(i+1)*3])
            rows.append(row)
        
        grid_frame = np.vstack(rows)
        out.write(grid_frame)
    
    # 释放资源
    for cap in caps:
        cap.release()
    out.release()
    
    print(f"简化版对比视频已保存至: {output_path}")

if __name__ == '__main__':
    print("选择处理模式:")
    print("1. 完整版 (带详细标签)")
    print("2. 简化版")
    
    choice = input("请选择 (1 或 2): ").strip()
    
    if choice == '1':
        create_video_grid()
    else:
        create_simple_grid()