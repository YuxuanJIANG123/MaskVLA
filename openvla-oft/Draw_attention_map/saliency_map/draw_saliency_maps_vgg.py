import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import requests
from PIL import Image

#Using VGG-19 pretrained model for image classification

model = torchvision.models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
img = Image.open('openvla-oft/debug_images/test_image/cam2/output_0001.png') 

def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))

def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

'''
    Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
    => Y/(1/σ) follows Distribution(0,σ)
    => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
'''
def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)

def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))

X = preprocess(img)

# 将模型设置为评估模式
model.eval()

# 我们需要相对于输入图像找到梯度，因此需要在其上调用requires_grad_
X.requires_grad_()

'''
通过模型进行前向传递以获取分数,注意VGG-19模型在末尾不执行softmax,
我们也不需要softmax,我们需要分数,这对我们来说非常完美。
'''

scores = model(X)

# 获取对应于最大分数的索引以及最大分数本身。
score_max_index = scores.argmax()
score_max = scores[0, score_max_index]

'''
在score_max上执行backward函数，将在计算图中执行反向传递，并计算score_max相对于计算图中节点的梯度
'''
score_max.backward()

'''
现在显著性将是相对于输入图像的梯度。但是请注意输入图像有3个通道
即红色R、绿色G和蓝色B。为了为每个像素i, j派生单一类别的显著性值
我们在所有颜色通道上取最大幅值。
'''
saliency, _ = torch.max(X.grad.data.abs(), dim=1)

# 保存显著性图
output_path = "/home/Better-oft/openvla-oft/Draw_attention_map/My_image/saliency_map.png"  # 保存路径
plt.imsave(output_path, saliency[0].cpu().numpy(), cmap='hot') 