from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@dataclass
class ViT3d_config:
    n_embd: int = 384
    n_heads: int = 12
    n_layers: int = 6
    n_channels: int = 1
    cnn_dim: int = 128
    total_patchs: int = 1200
    n_out_dim: int = 512


class Attention_3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.to_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.to_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.to_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.scale = config.n_embd**-0.5
        self.heads = config.n_heads
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        if self.flash:
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=False
            )
        else:
            print("using slow version")
            attn = q @ k.transpose(-2, -1) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = attn @ v
        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))


class CNN_embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(
                in_channels=config.n_channels,
                out_channels=config.cnn_dim,
                kernel_size=4,
                stride=2,
            ),
            nn.BatchNorm3d(config.cnn_dim),
            nn.ReLU(inplace=True),
        )
        self.conv1_3d = nn.Sequential(
            nn.Conv3d(
                in_channels=config.cnn_dim,
                out_channels=config.cnn_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(config.cnn_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2_3d = nn.Sequential(
            nn.Conv3d(
                in_channels=config.cnn_dim,
                out_channels=config.cnn_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm3d(config.cnn_dim),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=4, stride=2)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1_3d(x)
        x = self.conv2_3d(x)
        x = self.maxpool(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.FFN = FFN(config)
        self.Attention_3d = Attention_3d(config)
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.Attention_3d(self.norm1(x))
        x = x + self.FFN(self.norm2(x))
        return x


class ViT_3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.n_embd**-0.5
        self.to_cnn_embedding = nn.Sequential(
            CNN_embedding(config),
            Rearrange("b c d h w -> b (d h w) c"),
            nn.Linear(config.cnn_dim, config.n_embd),
        )
        self.positional_embedding = nn.Parameter(
            torch.empty(1, config.total_patchs + 1, config.n_embd)
        )
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.01)
        self.cls_token = nn.Parameter(self.scale * torch.randn(1, 1, config.n_embd))
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.to_out = nn.Linear(config.n_embd, config.n_out_dim)
        self.final_norm = RMSNorm(config.n_embd)
        self.apply(self.initialize_parameters)

    def initialize_parameters(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02, mean=0.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif hasattr(module, "weight"):
            nn.init.normal_(module.weight, std=0.02, mean=0.0)

    def forward(self, x):
        x = self.to_cnn_embedding(x)
        B, T, C = x.size()
        cls_tokens = repeat(self.cls_token, "1 1 d-> b 1 d", b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, : x.size(1)]
        for block in self.layers:
            x = block(x)
        x = self.final_norm(x)
        local_features = x
        global_features = self.to_out(x[:, 0, :])
        return global_features, local_features
