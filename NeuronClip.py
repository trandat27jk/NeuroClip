import inspect
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from dataloader import tokenizer
from Resnet_encoder import ResNet3D_18, VisionConfig
from text_encoder import RMSNorm, Text_alignment, TextConfig_1
from text_generation import Text_generation, TextConfig_2


@dataclass
class Neuron_config:
    n_embd: int = 512
    num_channels: int = 128


def take_local_features(local_image_features, num_channels):
    indices = torch.randperm(local_image_features.size(1))[:num_channels]
    selected_tensor = local_image_features[:, indices, :]
    return selected_tensor


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class NeuronClip(nn.Module):
    def __init__(
        self,
        image_encoder: callable,
        VisionConfig: VisionConfig,
        text_encoder_1: callable,
        TextConfig_1: TextConfig_1,
        text_encoder_2: callable,
        TextConfig_2: TextConfig_2,
        Neuron_config: Neuron_config,
        tokenizer,
    ):
        super().__init__()
        self.visual = image_encoder(VisionConfig)
        self.global_text_encoder = text_encoder_1(TextConfig_1)
        self.text_encoder_2 = text_encoder_2(TextConfig_2)
        self.text_projection = nn.Parameter(
            torch.empty(TextConfig_1.n_embd, Neuron_config.n_embd)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.Neuron_config = Neuron_config
        self.local_visual_projection = nn.Parameter(
            torch.empty(VisionConfig.out_channels, Neuron_config.n_embd)
        )

        self.local_texual_projection = nn.Parameter(
            torch.empty(TextConfig_1.n_embd, Neuron_config.n_embd)
        )
        self.TextConfig_1 = TextConfig_1
        self.TextConfig_2 = TextConfig_2
        self.to_logits = nn.Linear(TextConfig_2.n_embd, TextConfig_2.vocab_size)
        self.tokenizer = tokenizer
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_projection, std=self.TextConfig_1.n_embd**-0.5)
        nn.init.normal_(self.local_visual_projection, std=0.02, mean=0.0)
        nn.init.normal_(self.local_texual_projection, std=0.02, mean=0.0)
        nn.init.normal_(self.to_logits.weight, std=0.02, mean=0.0)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def image_encoder(self, image):
        return self.visual(image.type(self.dtype))

    def text_encoder_1(self, text):
        x = self.global_text_encoder(text).type(self.dtype)
        local_text_features = x[:, :-1, :]
        global_text_features = x[torch.arange(x.shape[0]), -1] @ self.text_projection
        return global_text_features, local_text_features

    def global_alignment(self, image, text):
        global_image_features, local_image_features = self.image_encoder(image)
        global_text_features, local_text_features = self.text_encoder_1(text)
        global_image_features = global_image_features / global_image_features.norm(
            dim=1, keepdim=True
        )
        global_text_features = global_text_features / global_text_features.norm(
            dim=1, keepdim=True
        )
        logit_scale = self.logit_scale.exp()
        logits_per_image = (
            logit_scale * global_image_features @ global_text_features.t()
        )
        logits_per_text = logits_per_image.t()

        # constrative loss
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        global_loss = (image_loss + text_loss) / 2

        return (
            global_image_features,
            global_text_features,
            local_image_features,
            local_text_features,
            global_loss,
        )

    def local_alignment(self, local_image_features, text_features, target_text):

        local_features_flattened = rearrange(
            local_image_features, "b c d h w -> b (d h w) c"
        )
        local_features_selected = take_local_features(
            local_features_flattened, self.Neuron_config.num_channels
        )
        local_features_selected = local_features_selected @ self.local_visual_projection

        text_features = text_features @ self.local_texual_projection
        combined_inputs = torch.cat((local_features_selected, text_features), dim=1)
        output = self.text_encoder_2(combined_inputs)
        logits = self.to_logits(output)
        if target_text is not None:
            logits_text = logits[:, -TextConfig_2.text_size :, :]
            local_loss = F.cross_entropy(
                logits_text.reshape(-1, logits_text.size(-1)),
                target_text.view(-1),
                ignore_index=1,
            )
        else:
            logits = logits[:, [-1], :]
            local_loss = None

        return logits, local_loss

    def configure_optimizers(self, weight_decay, learning_rates, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad()}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 1]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decay params {num_decay_params}")
        print(f"num nodecay params {num_nodecay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rates, betas=betas, **extra_args
        )
        print(f"Using fused AdamW :{use_fused}")
        return optimizer

    @torch.no_grad()
    def text_generation(self, idx, temperature, topk, max_tokens, image_path, device):
        idx = self.tokenizer(idx, return_tensors="pt")["input_ids"].to(device)
        image = np.expand_dims(np.load(image_path), axis=0)
        image = image / np.max(image)
        image = np.nan_to_num(image, copy=False)
        image_tensor = torch.tensor(image).unsqueeze(0).to(device)
        _, image_features = self.image_encoder(image_tensor)
        for i in range(max_tokens):
            idx_cond = (
                idx
                if idx.size(1) < self.TextConfig_2.text_size
                else idx[:, -self.TextConfig_2 :]
            )

            _, text_features = self.text_encoder_1(idx_cond)
            logits, _ = self.local_alignment(
                image_features, text_features, target_text=None
            )
            logits = logits[:, -1, :] / temperature
            if topk is not None:
                v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx_cond, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def text_generation_top_p(self, idx, temperature, top_p, max_tokens, image_path):
        idx = self.tokenizer(idx, return_tensors="pt")
        image = np.expand_dims(np.load(image_path), axis=0)
        image = image / np.max(image)
        image = np.nan_to_num(image, copy=False)
        image_tensor = torch.tensor(image)
        image_features = self.image_encoder(image_tensor)
        for i in range(max_tokens):
            idx_cond = (
                idx
                if idx.size(1) < self.TextConfig_2.text_size
                else idx[:, -self.TextConfig_2 :]
            )
            _, text_features = self.text_encoder_1(idx_cond)
            logits, _ = self.local_alignment(
                image_features, text_features, target_text=None
            )
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = sample_top_p(probs, top_p)
            idx = torch.cat((idx_cond, idx_next), dim=1)
        return idx
