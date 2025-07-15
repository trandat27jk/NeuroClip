import os

import hydra
import torch
from omegaconf import DictConfig

from dataloader import tokenizer
from NeuronClip import Neuron_config, NeuronClip
from Resnet_encoder import ResNet3D_18, VisionConfig
from text_encoder import RMSNorm, Text_alignment, TextConfig_1
from text_generation import Text_generation, TextConfig_2


def get_train_objs(
    Neuron_config, TextConfig_1, TextConfig_2, image_config, trainer_config
):
    model = NeuronClip(
        ResNet3D_18,
        image_config,
        Text_alignment,
        TextConfig_1,
        Text_generation,
        TextConfig_2,
        Neuron_config,
        tokenizer,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = model.configure_optimizers(
        trainer_config.learning_rates, betas=trainer_config.betas, device_type=device
    )
    return model, optimizer


@hydra.main(version_base=None, config_path=".", config_name="Neuron_clip_config")
def main(cfg: DictConfig):
    image_config = VisionConfig(**cfg["Vision_config"])
    TextConfig_1 = TextConfig_1(**cfg["Textconfig_1"])
