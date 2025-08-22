# 版权声明和许可证信息
# 注意：原始代码位于 https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import os  # 导入操作系统接口模块
import sys  # 导入系统模块
import argparse  # 导入命令行解析模块
import cv2  # 导入OpenCV库
import random  # 导入随机数生成模块
import colorsys  # 导入颜色系统转换模块
import requests  # 导入HTTP库
from io import BytesIO  # 导入用于处理输入输出的模块

import skimage.io  # 导入用于图像操作的scikit-image库
from skimage.measure import find_contours  # 导入用于寻找图像轮廓的函数
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
from matplotlib.patches import Polygon  # 导入用于绘制多边形的类
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入处理图像的模块
from torchvision import transforms as pth_transforms  # 导入预处理模块
import numpy as np  # 导入NumPy库
from PIL import Image  # 导入PIL库处理图像
from models.vision_transformer import vit_small, vit_large  # 导入Dinov2中的Vision Transformer模型
import types


# 主程序入口
if __name__ == '__main__':
    image_size = (952, 952) # 设置图像大小
    output_dir = '/home/Better-oft/openvla-oft/Draw_attention_map/dino_visual_attn/dinov2/output_1'  # 设置输出目录
    patch_size = 14  # 设置patch的大小

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设置设备为GPU或CPU

    # 初始化模型
    model = vit_large(
        patch_size=14,
        img_size=526,
        init_values=1.0,
        # ffn_layer="mlp",  # 可以选择前馈层的类型，这里被注释掉了
        block_chunks=0
    )

    # 加载模型权重
    model_path="/home/Better-oft/openvla-oft/Draw_attention_map/dino_visual_attn/dinov2/dinov2_vitl14_pretrain.pth"
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.load_state_dict(torch.load(model_path), strict=False)
    for p in model.parameters():
        p.requires_grad = False  # 冻结模型参数，不进行梯度更新
    model.to(device)  # 将模型移动到指定的设备
    model.eval()  # 设置模型为评估模式
    # print(model)

    print(hasattr(model, 'get_last_self_attention'))  # 输出 True
    # 加载并处理图像
    img = Image.open('/home/Better-oft/openvla-oft/Draw_attention_map/dino_visual_attn/dinov2/output_1/output_0024.png')  # 打开图像文件
    img = img.convert('RGB')  # 转换图像为RGB格式
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),  # 重设图像大小
        pth_transforms.ToTensor(),  # 将图像转换为Tensor
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 归一化处理
    ])
    img = transform(img)  # 应用变换
    print(img.shape)

    # 使图像尺寸适配patch大小
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)  # 调整图像尺寸并增加一个批次维度

    # 计算特征图的宽度和高度
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    print(img.shape)

    # 获取模型最后一层的注意力分数
    attentions = model.get_last_self_attention(img.to(device))

    nh = attentions.shape[1]  # 获取头部数量
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  # 重塑注意力分数
    print(torch.max(attentions, dim=1))  # 打印最大注意力值
    attentions[:, 283] = 0  # 将特定像素的注意力值设为0

    attentions = attentions.reshape(nh, w_featmap, h_featmap)  # 重塑注意力图
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()  # 上采样注意力图并转为numpy数组

    # 保存注意力热图
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")  # 设置文件名
        plt.imsave(fname=fname, arr=attentions[j], format='png')  # 保存热图
        print(f"{fname} saved.")  # 打印保存信息
        