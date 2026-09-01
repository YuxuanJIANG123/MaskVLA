import torch
import torch.nn as nn
from typing import Dict, Tuple
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
    )
from test_load_ckpt import (
load_vision_backbone_and_image_transform ,
get_gt_feature,
siglip_contrastive_loss,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
import torch.nn.functional as F
from test_load_ckpt import (
    load_vision_backbone_and_image_transform ,
    get_gt_feature,
    siglip_contrastive_loss,
)
import numpy as np

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)  # 生成注意力分数的单层网络
        self.query = self.query.to(torch.bfloat16)  # 确保权重是BFloat16

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, N, D] (N是token数量，D是特征维度)
        Returns:
            pooled: 池化后的特征 [B, D]
            weights: 注意力权重 [B, N]（可选，用于可视化）
        """
        # 计算注意力分数（未归一化）
        attn_scores = self.query(x).squeeze(-1)  # [B, N]
        
        # Softmax归一化为权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N]
        
        # 加权求和
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, attn_weights
    
class VLALossModel(nn.Module):
    def __init__(self, 
                 vla,
                #  vision_backbone,
                #  image_transform,
                 action_head,
                 num_patches,
                 siglip_loss_weight: float = 0.3,
                 dinov2_loss_weight: float = 0.2,
                 v2t_loss_weight: float = 0.1,
                 use_l1_regression: bool = True,
                 use_diffusion: bool = False,
                 use_proprio: bool = False,
                 ground_truth_actions = None,
                 output = None
                 ):
        """
        Loss model for VLA training that combines action prediction loss with visual feature losses.
        
        Args:
            vision_backbone: Pretrained vision backbone for ground truth feature extraction
            image_transform: Image transformation pipeline
            action_head: Action prediction head
            num_patches: Number of vision patches in the input
            siglip_loss_weight: Weight for SIGLIP feature matching loss
            dinov2_loss_weight: Weight for DINOv2 feature matching loss
            use_l1_regression: Whether to use L1 regression for action prediction
            use_diffusion: Whether diffusion is being used
            use_proprio: Whether proprioceptive state is used
        """
        super().__init__()
        self.vla=vla
        # self.vision_backbone = vision_backbone
        # self.image_transform = image_transform
        self.action_head = action_head
        self.num_patches = num_patches
        self.siglip_loss_weight = siglip_loss_weight
        self.dinov2_loss_weight = dinov2_loss_weight
        self.v2t_loss_weight = v2t_loss_weight
        self.use_l1_regression = use_l1_regression
        self.use_diffusion = use_diffusion
        self.use_proprio = use_proprio
        self.ground_truth_actions = ground_truth_actions
        self.output = output
        # 新增投影头（假设共享空间维度为256）
        self.vision_proj = nn.Linear(2176, 256)  # SigLIP特征维度1152 → 256
        self.text_proj = nn.Linear(4096, 256)    # 语言特征维度4096 → 256

        # Loss functions
        self.l1_loss = nn.L1Loss()

    def vision_language_contrastive_loss(self, vision_emb, text_emb, temperature=0.07):
        """CLIP风格的视觉-语言对比损失"""
        # 归一化特征
        vision_emb = F.normalize(vision_emb, dim=-1)  # [B, D]
        text_emb = F.normalize(text_emb, dim=-1)      # [B, D]
        
        # 计算相似度矩阵
        logits = (vision_emb @ text_emb.T) / temperature
        
        # 对称对比损失
        labels = torch.arange(len(logits), device=logits.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        return (loss_v2t + loss_t2v) / 2

    def action_language_alignment_loss(self, action_emb, text_emb):
        """动作-语言对齐损失（余弦相似度）"""
        action_emb = F.normalize(action_emb, dim=-1)  # [B, D]
        text_emb = F.normalize(text_emb, dim=-1)      # [B, D]
        return -torch.mean((action_emb * text_emb).sum(dim=-1))  # 最大化相似度

    def get_gt_visual_features(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract ground truth visual features from input images.
        
        Args:
            pixel_values: Input images tensor (B, C, H, W)
            
        Returns:
            Tuple of (combined_features, siglip_features, dinov2_features)
        """
        # print(f"Pixel values shape: {my_pixel_values.shape}")  
        all_combined_patches = []
        all_siglip_patches = []
        all_dinov2_patches = []
        #print(f"[DEBUG]the type of vla is", type(self.vla))#[INFO]the type of vla is <class 'torch.nn.parallel.distributed.DistributedDataParallel'>
        # print(f"[INFO]the content of vla is", self.vla)
        # print(f"[DEBUG] All attributes/methods of self.vla:", dir(self.vla))
        images = torch.split(pixel_values, [6] * self.vla.module.vision_backbone.get_num_images_in_input(), dim=1)
        for i, img in enumerate(images):  # 使用 enumerate 获取索引
            # Split each image further into two stacks of channels (each with 3 channels)
            img_regular, img_fused = torch.split(img.float(), [3, 3], dim=1)
            gt_siglip_feature, gt_dinov2_feature = get_gt_feature(self.vision_backbone,img_regular,img_fused)
            # print(f"Processing image {i} with shape: {img_regular.shape} and {img_fused.shape}")#Processing image with shape: torch.Size([4, 3, 224, 224]) and torch.Size([4, 3, 224, 224])
            # print(f"Siglip feature shape: {gt_siglip_feature.shape}")  # (B, num_patches * 256, D) torch.Size([2, 256, 1152])
            # print(f"Dinov2 feature shape: {gt_dinov2_feature.shape}")  # (B, num_patches * 256, D) torch.Size([2, 256, 1024])
            combined_patches = torch.cat([gt_dinov2_feature, gt_siglip_feature], dim=2)
            all_combined_patches.append(combined_patches)
            all_siglip_patches.append(gt_siglip_feature)
            all_dinov2_patches.append(gt_dinov2_feature)
            # # 保存常规图像（文件名：regular.png）
            # torchvision.utils.save_image(
            #     img_regular, 
            #     f"/home/jyx/MaskVLA/debug_images/batch_image/regular_{i}.png",
            #     normalize=True,  # 自动归一化到 [0, 1]
            # )
            
            # # 保存融合图像（文件名：fused.png）
            # torchvision.utils.save_image(
            #     img_fused, 
            #     f"/home/jyx/MaskVLA/debug_images/batch_image/fused_{i}.png",
            #     normalize=True,
            # )
                    
        # Concatenate along the patch dimension (dim=1)
        gt_combined_feature = torch.cat(all_combined_patches, dim=1)     # [B, N_total, 2D] torch.Size([2, 768, 2176])
        gt_siglip_feature = torch.cat(all_siglip_patches, dim=1)    # [B, N_total, D] torch.Size([2, 768, 1152])
        gt_dinov2_feature = torch.cat(all_dinov2_patches, dim=1)    # [B, N_total, D] torch.Size([2, 768, 1024])

        return gt_combined_feature, gt_siglip_feature, gt_dinov2_feature
    

    def vision_aligned_contrastive_loss(self,vision_hidden_states, actions_hidden_states, temperature=0.1, print_sim=False):
        """
        计算基于vision相似度的actions对比损失，并可选打印相似度矩阵
        
        参数:
            vision_hidden_states: 形状为 [batch_size, seq_len_v, hidden_size] 的视觉特征
            actions_hidden_states: 形状为 [batch_size, seq_len_a, hidden_size] 的动作特征
            temperature: 温度参数，控制相似度分布的尖锐程度
            print_sim: 是否打印相似度矩阵
            
        返回:
            loss: 对比损失值
            vision_sim: vision相似度矩阵 (detached)
            actions_sim: actions相似度矩阵 (detached)
        """
        batch_size = vision_hidden_states.shape[0]
        
        # 1. 计算vision序列的平均表示
        vision_avg = vision_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # 2. 计算vision之间的相似度矩阵 (batch内)
        vision_sim = F.cosine_similarity(
            vision_avg.unsqueeze(1),  # [batch_size, 1, hidden_size]
            vision_avg.unsqueeze(0),  # [1, batch_size, hidden_size]
            dim=-1
        ) / temperature  # [batch_size, batch_size]
        
        # 3. 计算actions序列的平均表示
        actions_avg = actions_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # 4. 计算actions之间的相似度矩阵
        actions_sim = F.cosine_similarity(
            actions_avg.unsqueeze(1),  # [batch_size, 1, hidden_size]
            actions_avg.unsqueeze(0),  # [1, batch_size, hidden_size]
            dim=-1
        ) / temperature  # [batch_size, batch_size]
        
        # 5. 可选打印相似度矩阵
        if print_sim:
            print("\nVision Similarity Matrix (after temperature scaling):")
            print(vision_sim.detach().cpu().round(decimals=4))  # PyTorch原生round
            
            print("\nActions Similarity Matrix (after temperature scaling):")
            print(actions_sim.detach().cpu().round(decimals=4))  # PyTorch原生round

            
            # 打印对角线元素（样本自身的相似度）
            print("\nDiagonal (self-similarity):")
            print("Vision:", vision_sim.diag().detach().cpu().round(decimals=4))
            print("Actions:", actions_sim.diag().detach().cpu().round(decimals=4))
        
        # 6. 计算KL散度损失
        loss = F.kl_div(
            F.log_softmax(actions_sim, dim=-1),
            F.softmax(vision_sim.detach(), dim=-1),  # 使用detach()防止梯度传播到vision分支
            reduction='batchmean',
            log_target=False
        )
        #print("[DEBUG] KL loss:", loss.item())

        return loss
    """
    Vision Similarity Matrix (after temperature scaling):
    tensor([[10.0000,  9.2500,  8.6250,  8.3750],
            [ 9.2500, 10.0625,  8.6875,  8.4375],
            [ 8.6250,  8.6875,  9.9375,  8.0625],
            [ 8.3750,  8.4375,  8.0625,  9.9375]], dtype=torch.bfloat16)

    Actions Similarity Matrix (after temperature scaling):
    tensor([[ 9.9375,  5.9688,  5.2500,  4.5000],
            [ 5.9688, 10.0625,  5.9375,  5.4688],
            [ 5.2500,  5.9375,  9.9375,  5.7500],
            [ 4.5000,  5.4688,  5.7500, 10.0625]], dtype=torch.bfloat16)

    Diagonal (self-similarity):
    Vision: tensor([10.0000, 10.0625,  9.9375,  9.9375], dtype=torch.bfloat16)
    Actions: tensor([ 9.9375, 10.0625,  9.9375, 10.0625], dtype=torch.bfloat16)
    [DEBUG] KL loss: 0.828125
    [DEBUG] align_loss has been calculated
    """
    def print_matrix(self, matrix, decimals=4):
        """支持BFloat16的矩阵打印方法"""
        # 先将BFloat16转换为Float32
        if matrix.dtype == torch.bfloat16:
            matrix = matrix.float()
        # 转换为numpy数组
        np_matrix = matrix.detach().cpu().numpy()
        # 格式化打印
        for row in np_matrix:
            print(" ".join([f"{x:.{decimals}f}" for x in row]))

    def clip_style_contrastive_loss(
        self,
        vision_features,  # shape: [batch_size, seq_len, hidden_dim]
        action_features,  # shape: [batch_size, seq_len, hidden_dim]
        temperature=0.07,
        pooling_method='attention',  # 新增：支持多种池化方式
        print_matrix=False
    ):
        """
        CLIP风格的对称对比损失（改进池化方法）
        支持池化方法：
        - 'mean': 平均池化
        - 'cls': 取第一个token（类似ViT）
        - 'attention': 注意力池化（类似CLIP的AttentionPool2d）
        """
        def pool(features, method):
            if method == 'mean':
                return features.mean(dim=1)  # [B,D]
            elif method == 'cls':
                return features[:, 0, :]     # 取CLS token
            elif method == 'attention':
                # 简化版注意力池化（类似CLIP）
                query = features.mean(dim=1, keepdim=True)  # [B,1,D]
                attn_weights = torch.softmax(
                    (features @ query.transpose(1,2)) / (features.shape[-1]**0.5),
                    dim=1
                )  # [B,T,1]
                return (attn_weights * features).sum(dim=1)  # [B,D]
            else:
                raise ValueError(f"不支持的池化方法: {method}")

        # 1. 池化处理
        vision_pooled = pool(vision_features, pooling_method)  # [B,D]
        action_pooled = pool(action_features, pooling_method)  # [B,D]

        # 2. 特征归一化（关键步骤）
        vision_norm = F.normalize(vision_pooled, p=2, dim=-1)
        action_norm = F.normalize(action_pooled, p=2, dim=-1)

        # 3. 计算相似度矩阵
        sim_matrix = vision_norm @ action_norm.T  # [B,B]
        logits = sim_matrix / temperature

        # 4. 计算对称损失
        batch_size = vision_norm.shape[0]
        labels = torch.arange(batch_size, device=vision_norm.device)
        loss_v2a = F.cross_entropy(logits, labels)  # vision->action
        loss_a2v = F.cross_entropy(logits.T, labels)  # action->vision
        loss = (loss_v2a + loss_a2v) / 2

        # 5. 打印调试信息
        if print_matrix:
            print("\n===== 池化调试信息 =====")
            print(f"池化方法: {pooling_method}")
            print(f"输入形状: vision={vision_features.shape}, action={action_features.shape}")
            print(f"池化后形状: vision={vision_pooled.shape}, action={action_pooled.shape}")

            self.print_matrix(sim_matrix.detach())

            print(f"\nLoss: vision->action={loss_v2a.item():.4f}, action->vision={loss_a2v.item():.4f}")
            print("=" * 50)

        return loss
    """
    ===== 池化调试信息 =====
    池化方法: attention
    输入形状: vision=torch.Size([4, 769, 4096]), action=torch.Size([4, 350, 4096])
    池化后形状: vision=torch.Size([4, 4096]), action=torch.Size([4, 4096])
    0.3164 0.2676 0.2852 0.2988
    0.3164 0.2695 0.2852 0.2988
    0.3359 0.2949 0.3203 0.3242
    0.3262 0.2793 0.2949 0.3184

    Loss: vision->action=1.3594, action->vision=1.3438

    Loss: vision->action=0.7148, action->vision=0.7109

    Loss: vision->action=1.0547, action->vision=1.0156

    Loss: vision->action=0.3848, action->vision=0.3867

    Loss: vision->action=0.4023, action->vision=0.2734

    Loss: vision->action=0.1348, action->vision=0.1250

    Loss: vision->action=0.0649, action->vision=0.0664

    Loss: vision->action=0.1201, action->vision=0.1260

    Loss: vision->action=0.0161, action->vision=0.0146

    Loss: vision->action=0.0173, action->vision=0.0188

    Loss: vision->action=0.0173, action->vision=0.0188

    Loss: vision->action=0.0093, action->vision=0.0084

    0.8594 0.3984 0.3809 0.4473
    0.4082 0.8594 0.4531 0.3672
    0.3984 0.4492 0.8711 0.4375
    0.4473 0.3457 0.4199 0.8672
    ==================================================
    """

    def forward(self, 
                vla_output,
                batch: Dict[str, torch.Tensor],
                device_id: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute forward pass and all losses.
        
        Args:
            vla_output: Output from VLA model forward pass
            batch: Input batch dictionary
            device_id: Device to use
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        ## jyx修改
        metrics = {}
        
        # print("Keys in batch:", batch.keys())
        my_language_instruction=batch["language_instructions"]
        # print(f"Language instruction: {my_language_instruction}")#Language instruction: ['pick up the banana and put it in the basket.', 'pick up the banana and put it in the basket.', 'pick up the banana and put it in the basket.', 'pick up the banana and put it in the basket.'] 
        
        my_pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id)
        # print(f"Pixel values shape: {my_pixel_values.shape}") 
        # 
        # gt vision feature 
        # gt_combined_feature, gt_siglip_feature, gt_dinov2_feature=self.get_gt_visual_features(my_pixel_values)
        # my_siglip_features = self.output.siglip_features  # (B, num_patches * 256, D)
        # my_dinov2_features = self.output.dinov2_features  # (B, num_patches * 256, D)
        # combined_vision_features = torch.cat([my_dinov2_features, my_siglip_features], dim=2)
        
        my_language_embeddings= self.output.language_embeddings  # (B, lang_seq_len, llm_dim) ([4, 28, 4096])
        # print(f"GT combined features shape: {gt_combined_feature.shape}")  # (B, num_patches * 256, 2D) ([4, 768, 2176])
        # print(f"GT Siglip features shape: {gt_siglip_feature.shape}")  #
        # print(f"GT Dinov2 features shape: {gt_dinov2_feature.shape}")  # (B, num_patches * 256, D) ([4, 768, 1152])
        # print(f"My Siglip features shape: {my_siglip_features.shape}")  # (bsz, 256 * num_images, D) ([4, 768, 1024])
        # print(f"My Dinov2 features shape: {my_dinov2_features.shape}")  # (bsz, 256 * num_images, D)  ([4, 768, 1152])
        # print(f"My Language embeddings shape: {my_language_embeddings.shape}") # (B, lang_seq_len, llm_dim) ([4, 28, 4096])

        # Get action masks needed for logging
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        # Get last layer hidden states取出最后一层 hidden states
        last_hidden_states = self.output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        # 输入是 [vision_patch_tokens] + [text_prompt_tokens] + [action_tokens]
        # 从第 num_patches 个 token 到倒数第 2 个 token 的隐藏状态
        # 跳过 vision token 和 action_tokens 只保留文本响应部分
        text_hidden_states = last_hidden_states[:, self.num_patches:-1]
        # print(f"Text hidden states shape: {text_hidden_states.shape}")  # (B, seq_len - num_patches - 1, D)
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            #只要是当前动作或下一个动作 token，就提取出来？？
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        ## jyx 0813
        vision_hidden_states = last_hidden_states[:, :self.num_patches]
        #print("[DEBUG]vision_hidden_states shape",vision_hidden_states.shape)
        #([4, 769, 4096])
        # 生成 prompt 的掩码（非动作部分）
        prompt_mask = ~(current_action_mask | next_actions_mask)  # 取反动作掩码

        # 提取 prompt hidden states
        prompt_hidden_states = (
            text_hidden_states[prompt_mask]  # 扁平化选择
            .reshape(batch_size, -1, text_hidden_states.shape[-1])  # 恢复形状
            .to(torch.bfloat16)  # 保持精度一致
        )  # (B, num_prompt_tokens, D)([4, 27, 4096])
        # print("[DEBUG]prompt_hidden_states",prompt_hidden_states.shape)

        # align_loss = self.vision_aligned_contrastive_loss(
        #     vision_hidden_states, actions_hidden_states, temperature=0.1, print_sim=False
        # )
        align_loss = self.clip_style_contrastive_loss(
            vision_hidden_states, 
            actions_hidden_states,
            pooling_method='attention',
            print_matrix=True
        )


        print("[DEBUG] align_loss has been calculated")
        # siglip_l1_loss = torch.nn.L1Loss()(gt_siglip_feature, my_siglip_features)
        # dinov2_l1_loss = torch.nn.L1Loss()(gt_dinov2_feature, my_dinov2_features)

        # 全局平均池化
        # vision_emb=combined_vision_features.mean(dim=1)          
        # text_emb = my_language_embeddings.mean(dim=1)   # [B, 4096] 平均文本token

        # # 注意力池化层
        # vision_attn_pool = AttentionPooling(input_dim=2176).to(device_id)  # 视觉特征维度
        # text_attn_pool = AttentionPooling(input_dim=4096).to(device_id)    # 语言特征维度

        # vision_emb, _ = vision_attn_pool(combined_vision_features).to(device_id)  # [B, 2176]

        # text_emb, _ = text_attn_pool(my_language_embeddings).to(device_id)  # [B, 4096]

        # v2t_loss = self.vision_language_contrastive_loss(
        # self.vision_proj(vision_emb),  # 投影到共享空间
        # self.text_proj(text_emb)       # 投影到共享空间
        # )
        # cross_view_loss = compute_cross_view_loss(all_combined_patches)
        # logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
        # # 初始化可学习的参数
        # img2text_contrastive_loss = siglip_contrastive_loss(
        #     text_embeddings=my_language_embeddings,
        #     image_features=my_siglip_features,
        #     logit_scale=logit_scale,
        #     logit_bias=None  # 可以设为 None 如果不使用偏置
        # )

        # Predict action# 把hidden states reshape 并通过 MLP预测连续动作向量。
        # print(f"the type of action head is", type(self.action_head))
        predicted_actions = self.action_head.module.predict_action(actions_hidden_states)
        # Get full L1 loss
        action_loss = torch.nn.L1Loss()(self.ground_truth_actions, predicted_actions)
        # weight1 = self.siglip_loss_weight
        # weight2 = self.dinov2_loss_weight
        # weight3 = self.v2t_loss_weight
        # loss = action_loss + weight1 * siglip_l1_loss + weight2 * dinov2_l1_loss
        loss = action_loss + align_loss


        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
            }
        )


        ground_truth_curr_action = self.ground_truth_actions[:, 0]
        predicted_curr_action = predicted_actions[:, 0]
        ground_truth_next_actions = self.ground_truth_actions[:, 1:]
        predicted_next_actions = predicted_actions[:, 1:]
        curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
        next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
        # siglip_l1_loss = torch.nn.L1Loss()(gt_siglip_feature, my_siglip_features)
        # dinov2_l1_loss = torch.nn.L1Loss()(gt_dinov2_feature, my_dinov2_features)
        metrics.update(
            {
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                "align_loss":align_loss.item(),
                # "siglip_l1_loss": siglip_l1_loss.item(),
                # "dinov2_l1_loss": dinov2_l1_loss.item(),
                # "v2t_contrastive_loss": v2t_loss.item(),
            }
        )

        return loss, metrics