import inspect
import math
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


@dataclass
class TextConfig_1:
    block_size: int = 148  # max sequence length
    vocab_size: int = 42384
    n_layer: int = 6  # number of layers
    n_head: int = 8  # number of heads
    n_embd: int = 512  # embedding dimension


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


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.to_qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.head = config.n_head
        self.n_embd = config.n_embd
        self.to_out = nn.Linear(config.n_embd, config.n_embd)
        self.rotary_embedding = RotaryEmbedding(config.n_embd // config.n_head)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("Using slow version")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.to_qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.head, C // self.head).transpose(1, 2)
        q = q.view(B, T, self.head, C // self.head).transpose(1, 2)
        v = v.view(B, T, self.head, C // self.head).transpose(1, 2)
        q = self.rotary_embedding.rotate_queries_or_keys(q)
        k = self.rotary_embedding.rotate_queries_or_keys(k)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.to_out(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Text_alignment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def configure_optimizers(self, weight_decay, learning_rate, device_type):
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    #     optim_groups = [
    #         {"params": decay_params, "weight_decay": weight_decay},
    #         {"params": nodecay_params, "weight_decay": 0.0},
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     print(f"number params decay: {num_decay_params}")
    #     print(f"number nodecay_params: {num_nodecay_params}")
    #     fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == "cuda"
    #     optimizer = torch.optim.AdamW(
    #         optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    #     )
    #     return optim4izer

    def forward(self, x):
        B, T = x.size()
        assert T <= self.config.block_size
        x = self.token_embedding(x)
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)
        return x
