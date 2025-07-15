import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import fsspec
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from NeuronClip import Neuron_config, NeuronClip
from text_encoder import Text_alignment, TextConfig_1
from text_generation import Text_generation, TextConfig_2


@dataclass
class TrainerConfig:
    batch_size: int = None
    use_amp: bool = None
    max_epochs: int = None
    learning_rate: float = None
    warmup_iters: int = None
    min_lr: float = None
    save_every: int = None
    alpha: float = None
    decay_lr: bool = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None


class Snapshot:
    model_state: "OrderedDict[str,torch.tensor]"
    optimizer_state: Dict[str, Any]
    finished_epoch: int


def save_file(name, value, epoch, mode="a"):
    with open(name, mode) as file:
        file.write(f"Epoch {epoch}, Loss: {value:.4f}\n")


class Trainer(nn.Module):
    def __init__(
        self, Trainerconfig, model, optimizer, train_dataset, test_dataset, train_test
    ):
        super().__init__()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.trainerconfig = Trainerconfig
        self.model = model.to(self.local_rank)
        self.train_loader = self.processing_data(train_dataset)
        self.test_loader = self.processing_data(test_dataset)
        self.train_test_loader = self.processing_data(train_test, train=False)
        if self.trainerconfig.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.optimizer = optimizer
        self.data_num_workers = int(os.cpu_count()) // 2
        self.epochs_run = 0
        if self.trainerconfig.snapshot_path is None:
            self.trainerconfig.snapshot_path = "snapshot.pt"
        self._load_snapshot()
        # self.model=DDP(self.model)

    def processing_data(self, dataset, train=False):
        return DataLoader(
            dataset,
            batch_size=self.trainerconfig.batch_size,
            shuffle=train,
            pin_memory=True,
        )

    def _run_batch(
        self, text_tokens, image, input_text, target_text, batch_idx, train=True
    ):
        lr = (
            self.get_lr(batch_idx)
            if self.trainerconfig.decay_lr
            else self.trainerconfig.learning_rate
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        with torch.set_grad_enabled(train), torch.autocast(
            "cuda", dtype=self.dtype, enabled=(self.trainerconfig.use_amp)
        ):
            (
                logits_per_image,
                logits_per_text,
                global_image_features,
                global_text_features,
                local_image_features,
                local_text_features,
                global_loss,
            ) = self.model.global_alignment(image, text_tokens)
            local_alignment_output, local_loss = self.model.local_alignment(
                local_image_features, local_text_features, target_text
            )
            loss = (
                self.trainerconfig.alpha * global_loss
                + (1 - self.trainerconfig.alpha) * local_loss
            )
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.trainerconfig.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainerconfig.grad_norm_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainerconfig.grad_norm_clip
                )
                self.optimizer.step()
            self.model.logit_scale.data = torch.clamp(
                self.model.logit_scale.data, 0, 4.6052
            )

        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        epoch_loss=0.0
        num_batchs=0.0
        for batch_idx, (text_tokens, image, input_text, target_text) in enumerate(
            dataloader
        ):
            text_tokens, image, input_text, target_text = (
                text_tokens.to(self.local_rank),
                image.to(self.local_rank),
                input_text.to(self.local_rank),
                target_text.to(self.local_rank),
            )
            loss = self._run_batch(text_tokens, image, input_text, target_text, train)
            epoch_loss+=loss
            num_batchs+=1
            if self.local_rank == 0:
                print(f"Runing  at iner: {batch_idx}, Epoch; {epoch}, loss: {loss}")
            return epoch_loss/num_batchs
    def get_lr(self, it):
        if it < self.trainerconfig.warmup_iters:
            return (
                self.trainerconfig.learning_rate
                * (it + 1)
                / (self.trainerconfig.warmup_iters + 1)
            )
        if it > self.trainerconfig.lr_decay_iters:
            return self.trainerconfig.min_lr
        decay_ratio = (it - self.trainerconfig.warmup_iters) / (
            self.trainerconfig.lr_decay_iters - self.trainerconfig.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.trainerconfig.min_lr + coeff * (
            self.trainerconfig.learning_rate - self.trainerconfig.min_lr
        )

    @torch.no_grad()
    def task_retrieval(self, test_dataloader, top_k):
        total_image_features = []
        total_text_features = []
        for batch_idx, (image, text) in enumerate(test_dataloader):
            image, text = image.to(self.local_rank), text.to(self.local_rank)
            (
                global_image_features,
                global_text_features,
                global_loss,
            ) = self.model(image, text)
            total_image_features.append(global_image_features)
            total_text_features.append(global_text_features)
        total_image_features = torch.cat(total_image_features)
        total_text_features = torch.cat(total_text_features)
        num_images, num_texts = len(total_image_features), len(total_text_features)
        assert num_images == num_texts, "Number of images and texts must match"

        # Image-to-Text
        measure_image_text = total_image_features @ total_text_features.t()
        top_k_values_image_text, top_k_indices_image_text = torch.topk(
            measure_image_text, k=top_k, dim=1
        )
        labels = torch.arange(num_images).to(self.local_rank)
        correct_image_text = (
            (top_k_indices_image_text == labels.unsqueeze(1)).any(dim=1).sum().item()
        )

        # Text-to-Image
        measure_text_image = measure_image_text.t()
        labels_text = torch.arange(num_texts).to(self.local_rank)
        top_k_values_text_image, top_k_indices_text_image = torch.topk(
            measure_text_image, k=top_k, dim=1
        )
        correct_text_image = (
            (top_k_indices_text_image == labels_text.unsqueeze(1))
            .any(dim=1)
            .sum()
            .item()
        )
        recall_image_text = correct_image_text / num_images
        recall_text_image = correct_text_image / num_texts

        return recall_image_text, recall_text_image

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.trainerconfig.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming at {self.epochs_run}")

    def _save_snapshot(self, epoch):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
        )

        torch.save(snapshot, self.config.snapshot_path)
        print(f"Snapshot saved at epoch {epoch}")

    def train(self):
        for epoch in range(self.epochs_run, self.trainerconfig.max_epochs):
            # measure loss train
            train_loss_epoch = self._run_epoch(epoch, self.train_loader, train=True)
            save_file("train_loss.txt", value=train_loss_epoch, epoch=epoch, mode="a")
            # measure loss val
            val_loss_epoch = self._run_epoch(epoch, self.test_loader, train=False)
            save_file("val_loss.txt", value=val_loss_epoch, epoch=epoch, mode="a")
            if epoch % self.trainerconfig.save_every == 0:
                self._save_snapshot(epoch)
                correct_image_text, correct_text_image = self.task_retrieval(
                    self.test_loader, top_k=1
                )
                print(
                    f"Retrieval task R@1 {correct_image_text} and {correct_text_image}"
                )
                save_file(
                    "image_text_test_top_1_retrieval.txt",
                    value=correct_image_text,
                    epoch=epoch,
                    mode="a",
                )
                save_file(
                    "text_image_test_top_1_retrieval.txt",
                    value=correct_text_image,
                    epoch=epoch,
                    mode="a",
                )
                correct_image_text_top_5, correct_text_image_top_5 = (
                    self.task_retrieval(self.test_loader, top_k=5)
                )
                save_file(
                    "image_text_test_top_5_retrieval.txt",
                    value=correct_image_text_top_5,
                    epoch=epoch,
                    mode="a",
                )
                save_file(
                    "text_image_test_top_5_retrieval.txt",
                    value=correct_text_image_top_5,
                    epoch=epoch,
                    mode="a",
                )
                print(
                    f"Retrieval task R@5 {correct_image_text_top_5} and {correct_text_image_top_5}"
                )
                correct_image_text_train, correct_text_image_train = (
                    self.task_retrieval(self.train_test_loader, top_k=1)
                )
                save_file(
                    "image_text_train_top_1_retrieval.txt",
                    value=correct_image_text_train,
                    epoch=epoch,
                    mode="a",
                )
                save_file(
                    "text_image_train_top_1_retrieval.txt",
                    value=correct_text_image_train,
                    epoch=epoch,
                    mode="a",
                )

                correct_image_text_train_top_5, correct_text_image_train_top_5 = (
                    self.task_retrieval(self.train_test_loader, top_k=5)
                )
                save_file(
                    "image_text_train_top_1_retrieval.txt",
                    value=correct_image_text_train_top_5,
                    epoch=epoch,
                    mode="a",
                )
                save_file(
                    "text_image_train_top_1_retrieval.txt",
                    value=correct_text_image_train_top_5,
                    epoch=epoch,
                    mode="a",
                )
