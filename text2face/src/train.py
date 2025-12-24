"""
训练脚本

Text2FaceKeypoints 模型训练
"""

import os
import argparse
import yaml
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

from src.models.model import Text2FaceKeypoints, Text2FaceKeypointsWithEmotion, get_model
from src.data.dataset import TextKeypointDataset, collate_fn


class KeypointLoss(nn.Module):
    """
    关键点损失函数

    包含:
        - Landmark MSE Loss
        - Pose MSE Loss
        - Temporal Consistency Loss (可选)
    """

    def __init__(
        self,
        landmark_weight: float = 1.0,
        pose_weight: float = 1.0,
        temporal_weight: float = 0.1,
        use_temporal: bool = False,
    ):
        super().__init__()
        self.landmark_weight = landmark_weight
        self.pose_weight = pose_weight
        self.temporal_weight = temporal_weight
        self.use_temporal = use_temporal

        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred_landmarks: torch.Tensor,
        pred_pose: torch.Tensor,
        gt_landmarks: torch.Tensor,
        gt_pose: torch.Tensor,
    ) -> dict:
        """
        计算损失

        Args:
            pred_landmarks: (B, T, 21, 3) or (B, 21, 3)
            pred_pose: (B, T, 6) or (B, 6)
            gt_landmarks: (B, T, 21, 3) or (B, 21, 3)
            gt_pose: (B, T, 6) or (B, 6)

        Returns:
            dict with total_loss and individual losses
        """
        # 关键点损失
        landmark_loss = self.mse_loss(pred_landmarks, gt_landmarks)

        # 姿态损失
        pose_loss = self.mse_loss(pred_pose, gt_pose)

        # 总损失
        total_loss = (
            self.landmark_weight * landmark_loss +
            self.pose_weight * pose_loss
        )

        # 时序一致性损失
        if self.use_temporal and pred_landmarks.dim() == 4:  # (B, T, 21, 3)
            temporal_loss = self._temporal_consistency(pred_landmarks, pred_pose)
            total_loss += self.temporal_weight * temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=pred_landmarks.device)

        return {
            "total": total_loss,
            "landmark": landmark_loss,
            "pose": pose_loss,
            "temporal": temporal_loss,
        }

    def _temporal_consistency(
        self,
        landmarks: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        时序一致性损失

        惩罚相邻帧之间的剧烈变化
        """
        # 计算相邻帧差分
        diff_lm = landmarks[:, 1:] - landmarks[:, :-1]  # (B, T-1, 21, 3)
        diff_pose = pose[:, 1:] - pose[:, :-1]  # (B, T-1, 6)

        # L2 范数
        consistency_loss = (
            torch.mean(diff_lm ** 2) +
            torch.mean(diff_pose ** 2)
        )

        return consistency_loss


class Trainer:
    """
    训练器
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: KeypointLoss,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: str,
        save_dir: str,
        log_interval: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self.best_val_loss = float("inf")
        self.global_step = 0

    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0
        total_landmark_loss = 0
        total_pose_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            gt_landmarks = batch["landmarks"].to(self.device)
            gt_pose = batch["pose"].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            pred_landmarks, pred_pose = self.model(
                input_ids,
                attention_mask,
                sequence_length=gt_landmarks.shape[1] if gt_landmarks.dim() == 3 else 1,
            )

            # 计算损失
            losses = self.criterion(pred_landmarks, pred_pose, gt_landmarks, gt_pose)
            loss = losses["total"]

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            total_landmark_loss += losses["landmark"].item()
            total_pose_loss += losses["pose"].item()

            self.global_step += 1

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lm": f"{losses['landmark'].item():.4f}",
                "pose": f"{losses['pose'].item():.4f}",
            })

            # 日志
            if batch_idx % self.log_interval == 0:
                self._log_loss(epoch, batch_idx, losses)

        # 平均损失
        num_batches = len(self.train_loader)
        return {
            "total": total_loss / num_batches,
            "landmark": total_landmark_loss / num_batches,
            "pose": total_pose_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """验证"""
        self.model.eval()

        total_loss = 0
        total_landmark_loss = 0
        total_pose_loss = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # 移动到设备
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            gt_landmarks = batch["landmarks"].to(self.device)
            gt_pose = batch["pose"].to(self.device)

            # 前向传播
            pred_landmarks, pred_pose = self.model(
                input_ids,
                attention_mask,
                sequence_length=gt_landmarks.shape[1] if gt_landmarks.dim() == 3 else 1,
            )

            # 计算损失
            losses = self.criterion(pred_landmarks, pred_pose, gt_landmarks, gt_pose)

            total_loss += losses["total"].item()
            total_landmark_loss += losses["landmark"].item()
            total_pose_loss += losses["pose"].item()

        # 平均损失
        num_batches = len(self.val_loader)
        return {
            "total": total_loss / num_batches,
            "landmark": total_landmark_loss / num_batches,
            "pose": total_pose_loss / num_batches,
        }

    def train(self, num_epochs: int):
        """完整训练流程"""
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_losses = self.train_epoch(epoch)

            # 验证
            val_losses = self.validate()

            # 更新学习率
            self.scheduler.step()

            # 打印
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss:   {val_losses['total']:.4f}")

            # 保存检查点
            self._save_checkpoint(epoch, val_losses["total"])

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """保存检查点"""
        is_best = val_loss < self.best_val_loss

        if is_best:
            self.best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
        }

        # 保存最新
        torch.save(checkpoint, self.save_dir / "checkpoint_latest.pt")

        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.save_dir / "checkpoint_best.pt")

        # 定期保存
        if epoch % 10 == 0:
            torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch}.pt")

    def _log_loss(self, epoch: int, batch_idx: int, losses: dict):
        """记录损失"""
        log_entry = {
            "epoch": epoch,
            "batch": batch_idx,
            "step": self.global_step,
            "loss": losses["total"].item(),
            "landmark_loss": losses["landmark"].item(),
            "pose_loss": losses["pose"].item(),
        }

        log_file = self.save_dir / "train_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Text2FaceKeypoints model")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")

    parser.add_argument("--train-data", type=str,
                        help="Path to training data (overrides config)")

    parser.add_argument("--val-data", type=str,
                        help="Path to validation data (overrides config)")

    parser.add_argument("--output", type=str,
                        help="Output directory (overrides config)")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    parser.add_argument("--resume", type=str,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行覆盖
    if args.train_data:
        config["data"]["train_path"] = args.train_data
    if args.val_data:
        config["data"]["val_path"] = args.val_data
    if args.output:
        config["output_dir"] = args.output

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Output directory: {config['output_dir']}")

    # 创建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])

    # 创建数据集
    print("Loading datasets...")

    train_dataset = TextKeypointDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        augment=config["data"].get("augment", True),
    )

    val_dataset = TextKeypointDataset(
        data_path=config["data"]["val_path"],
        tokenizer=tokenizer,
        augment=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 创建模型
    print("Creating model...")

    model = get_model(
        model_type=config["model"].get("type", "base"),
        **config["model"]
    )

    model = model.to(device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数
    criterion = KeypointLoss(
        landmark_weight=config["training"].get("landmark_weight", 1.0),
        pose_weight=config["training"].get("pose_weight", 1.0),
        temporal_weight=config["training"].get("temporal_weight", 0.1),
        use_temporal=config["model"].get("use_temporal", False),
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    # 学习率调度器
    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_scheduler(
        config["training"].get("scheduler", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=config["training"].get("warmup_steps", 500),
        num_training_steps=num_training_steps,
    )

    # 恢复训练
    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config["output_dir"],
        log_interval=config["training"].get("log_interval", 10),
    )

    # 训练
    print(f"\nStarting training from epoch {start_epoch}...")
    trainer.train(config["training"]["num_epochs"])

    print(f"\nTraining complete!")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
