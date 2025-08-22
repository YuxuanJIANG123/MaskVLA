# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import nn, Tensor


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    # def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    #     q, k, v = torch.unbind(qkv, 2)
    #     q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
    #     x = nn.functional.scaled_dot_product_attention(
    #         q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
    #     )
    #     x = x.transpose(1, 2).contiguous().view(B, N, C)
    #     x = self.proj_drop(self.proj(x))
    #     return x
    def forward(self, x: Tensor, return_attn=False) -> Tensor:
        # print("I am in Attention forward method")
        """
        处理输入张量，计算注意力得分并可选择返回它们。

        参数:
        x (Tensor): 输入特征数据，形状为(B, N, C)其中B是批量大小，N是序列长度，C是通道数。
        return_attn (bool): 如果为True，则返回注意力矩阵而不是正常的前向传播结果。
        """
        # 获取输入的维度
        B, N, C = x.shape
        # 计算查询、键、值（qkv）并调整其维度以适应多头注意力的需求
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 分离查询、键和值，对查询进行缩放
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        # 计算注意力得分
        attn = q @ k.transpose(-2, -1)

        # 对注意力得分进行softmax操作并应用dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算输出特征
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 如果return_attn为True，则返回注意力得分
        if return_attn:
            return attn

        # 否则返回处理后的输出特征
        return x


# class MemEffAttention(Attention):
#     def forward(self, x: Tensor, attn_bias=None) -> Tensor:
#         if not XFORMERS_AVAILABLE:
#             if attn_bias is not None:
#                 raise AssertionError("xFormers is required for using nested tensors")
#             return super().forward(x)

#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

#         q, k, v = unbind(qkv, 2)

#         x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
#         x = x.reshape([B, N, C])

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        """
        一种内存效率高的注意力机制，可选地返回注意力矩阵。

        参数:
        x (Tensor): 输入张量，形状为(B, N, C)，其中B是批次大小，N是序列长度，C是通道数。
        attn_bias (Tensor, 可选): 用于注意力的可选偏置张量，通常用于相对位置嵌入或掩蔽。
        return_attn (bool): 如果为True，返回注意力矩阵而不是正常的前向传播输出。

        返回:
        Tensor: 在注意力计算之后的输出张量，或者如果return_attn为True，则是注意力矩阵。
        """
        # print("I am in MemEffAttention forward method")
        # 检查是否安装了xFormers，以便使用像嵌套张量这样的高级功能
        if not XFORMERS_AVAILABLE:
            # 确保在xFormers不可用时，不使用attn_bias
            assert attn_bias is None, "使用嵌套张量需要安装xFormers"
            # 调用超类方法，并传递return_attn参数
            return super().forward(x, return_attn)

        # 分解输入尺寸
        B, N, C = x.shape
        # 计算查询、键和值张量
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # 将qkv张量分解为单独的查询、键和值张量
        q, k, v = unbind(qkv, 2)

        # 使用提供的或计算出的偏置应用内存效率高的注意力
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # 将输出投影回原始张量形状并应用dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x