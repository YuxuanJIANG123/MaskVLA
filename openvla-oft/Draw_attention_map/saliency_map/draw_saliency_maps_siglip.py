import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
import torchvision.transforms as T

# 加载SigLIP模型
model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')

# 定义文本标签
labels_list = ["a dog", "a cat", "a banana"]
text = tokenizer(labels_list, context_length=model.context_length)

# 加载图像
img_path = "/home/Better-oft/openvla-oft/debug_images/test_image/cam2/output_0001.png"
image = Image.open(img_path)

# 自定义预处理和后处理函数
def preprocess_image(image, size=384):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

def deprocess_image(tensor):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[2.0, 2.0, 2.0]),
        T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(tensor)

# 预处理图像
X = preprocess_image(image)
X.requires_grad_()

# 将模型设置为评估模式
model.eval()

# 前向传播
with torch.cuda.amp.autocast():
    image_features = model.encode_image(X)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算与每个文本的相似度
    logits_per_image = image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias
    probs = torch.sigmoid(logits_per_image)

# 获取最高概率的类别
score_max_index = probs.argmax()
score_max = probs[0, score_max_index]

# 反向传播计算梯度
score_max.backward()

# 获取显著性图
# saliency, _ = torch.max(X.grad.data.abs(), dim=1)

saliency = torch.max(X.grad.data.abs(), dim=1)[0].cpu().numpy()[0]
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
# # 保存显著性图
# output_path = "/home/Better-oft/openvla-oft/Draw_attention_map/My_image/siglip_saliency_map.png"
# plt.imsave(output_path, saliency[0].cpu().numpy(), cmap='hot') 
# 3. 保存结果
output_dir = "/home/Better-oft/openvla-oft/Draw_attention_map/My_image/"
resized_img = image.resize((384, 384))
# 3.1 纯热图
plt.imsave(f"{output_dir}siglip_heatmap.png", saliency, cmap='hot')

# 3.2 热图叠加原图
resized_img = image.resize((384, 384))
heatmap_img = Image.fromarray(np.uint8(plt.cm.hot(saliency)*255)[..., :3])  # 只取RGB通道
blended = Image.blend(resized_img.convert("RGB"), heatmap_img.convert("RGB"), alpha=0.5)
blended.save(f"{output_dir}siglip_overlay.png")

# 3.3 三合一对比图
composite = Image.new('RGB', (384 * 3, 384))
composite.paste(resized_img, (0, 0))
composite.paste(heatmap_img, (384, 0))
composite.paste(blended, (384 * 2, 0))
composite.save(f"{output_dir}siglip_comparison.png")

print(f"Saliency map saved to {output_dir}")
print("Label probabilities:", list(zip(labels_list, [round(p.item(), 3) for p in probs[0]])))