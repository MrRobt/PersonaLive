"""
数据集类

用于训练 Text2FaceKeypoints 模型的数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import random


class TextKeypointDataset(Dataset):
    """
    文本-关键点配对数据集

    数据格式:
        {
            "text": "开心的微笑",
            "landmarks": [[x1,y1,z1], ..., [x21,y21,z21]],  # (21, 3)
            "pitch": 0.1,
            "yaw": -0.2,
            "roll": 0.0,
            "t": [0, 0, 0],
            "scale": 1.0
        }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 128,
        augment: bool = True,
        temporal: bool = False,
        sequence_length: int = 8,
    ):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径 (.jsonl)
            tokenizer: 文本分词器
            max_length: 最大文本长度
            augment: 是否数据增强
            temporal: 是否时序数据
            sequence_length: 时序长度
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.temporal = temporal
        self.sequence_length = sequence_length

        # 加载数据
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """加载数据"""
        data = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)

        return data

    def __len__(self) -> int:
        if self.temporal:
            # 时序模式: 返回可能的序列数量
            return max(0, len(self.data) - self.sequence_length + 1)
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""

        if self.temporal:
            # 时序模式: 获取序列
            return self._get_sequence(idx)
        else:
            # 单帧模式
            return self._get_single(idx)

    def _get_single(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单帧样本"""
        item = self.data[idx]

        # 文本
        text = item["text"]

        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        # 关键点
        landmarks = torch.tensor(item["landmarks"], dtype=torch.float32)  # (21, 3)

        # 姿态
        pose = torch.tensor([
            item["pitch"],
            item["yaw"],
            item["roll"],
            item["t"][0],
            item["t"][1],
            item["scale"],
        ], dtype=torch.float32)  # (6,)

        # 数据增强
        if self.augment:
            landmarks, pose = self._augment(landmarks, pose)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "landmarks": landmarks,
            "pose": pose,
        }

    def _get_sequence(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取时序样本"""
        # 获取序列索引
        indices = list(range(idx, idx + self.sequence_length))

        # 收集序列数据
        items = [self.data[i] for i in indices]

        # 文本 (使用第一个样本的文本)
        text = items[0]["text"]

        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        # 关键点序列
        landmarks = torch.stack([
            torch.tensor(item["landmarks"], dtype=torch.float32)
            for item in items
        ])  # (T, 21, 3)

        # 姿态序列
        pose = torch.stack([
            torch.tensor([
                item["pitch"],
                item["yaw"],
                item["roll"],
                item["t"][0],
                item["t"][1],
                item["scale"],
            ], dtype=torch.float32)
            for item in items
        ])  # (T, 6)

        # 数据增强
        if self.augment:
            landmarks, pose = self._augment_sequence(landmarks, pose)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "landmarks": landmarks,
            "pose": pose,
        }

    def _augment(
        self,
        landmarks: torch.Tensor,
        pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """数据增强"""
        # 随机旋转
        if random.random() < 0.3:
            angle = random.uniform(-5, 5) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ], dtype=torch.float32)
            landmarks = landmarks @ rotation.T

        # 随机缩放
        if random.random() < 0.3:
            scale = random.uniform(0.95, 1.05)
            landmarks = landmarks * scale

        # 随机平移
        if random.random() < 0.3:
            shift = torch.randn(3) * 0.01
            landmarks = landmarks + shift

        # 姿态抖动
        if random.random() < 0.3:
            noise = torch.randn(6) * 0.02
            pose = pose + noise

        return landmarks, pose

    def _augment_sequence(
        self,
        landmarks: torch.Tensor,
        pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """时序数据增强"""
        # 时序平滑噪声
        if random.random() < 0.3:
            noise = torch.randn_like(landmarks) * 0.01
            landmarks = landmarks + noise

        if random.random() < 0.3:
            noise = torch.randn_like(pose) * 0.02
            pose = pose + noise

        return landmarks, pose


class TextKeypointDatasetWithEmotion(TextKeypointDataset):
    """
    带情绪标签的数据集

    额外字段:
        - emotion: 情绪标签 (0-6)
        - intensity: 情绪强度 [0, 1]
    """

    def _get_single(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单帧样本"""
        item = self.data[idx]

        # 文本
        text = item["text"]

        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        # 关键点
        landmarks = torch.tensor(item["landmarks"], dtype=torch.float32)

        # 姿态
        pose = torch.tensor([
            item["pitch"],
            item["yaw"],
            item["roll"],
            item["t"][0],
            item["t"][1],
            item["scale"],
        ], dtype=torch.float32)

        # 情绪
        emotion = torch.tensor(item.get("emotion", 0), dtype=torch.long)
        intensity = torch.tensor(item.get("intensity", 0.5), dtype=torch.float32)

        # 数据增强
        if self.augment:
            landmarks, pose = self._augment(landmarks, pose)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "landmarks": landmarks,
            "pose": pose,
            "emotion": emotion,
            "intensity": intensity,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    数据整理函数

    Args:
        batch: 样本列表

    Returns:
        整理后的批次数据
    """
    # 检查是否是时序数据
    is_temporal = batch[0]["landmarks"].dim() == 3

    if is_temporal:
        # 时序数据
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "landmarks": torch.stack([item["landmarks"] for item in batch]),
            "pose": torch.stack([item["pose"] for item in batch]),
        }
    else:
        # 单帧数据
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "landmarks": torch.stack([item["landmarks"] for item in batch]),
            "pose": torch.stack([item["pose"] for item in batch]),
        }


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    创建数据加载器

    Args:
        data_path: 数据文件路径
        tokenizer: 文本分词器
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **dataset_kwargs: 数据集参数

    Returns:
        DataLoader
    """
    dataset = TextKeypointDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        **dataset_kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    # 测试数据集
    print("Creating test dataset...")

    # 创建测试数据
    test_data = [
        {
            "text": "开心的微笑",
            "landmarks": np.random.randn(21, 3).tolist(),
            "pitch": 0.1,
            "yaw": -0.2,
            "roll": 0.0,
            "t": [0, 0, 0],
            "scale": 1.0,
        }
    ] * 100

    test_path = "test_data.jsonl"
    with open(test_path, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # 测试数据集
    dataset = TextKeypointDataset(test_path, tokenizer=None)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"  input_ids: {sample['input_ids'].shape}")
    print(f"  landmarks: {sample['landmarks'].shape}")
    print(f"  pose: {sample['pose'].shape}")

    # 清理
    Path(test_path).unlink()
