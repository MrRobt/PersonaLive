"""
数据集样本生成器

快速生成各种情绪和姿态的训练数据样本
"""

import json
import argparse
import numpy as np
from pathlib import Path


# 情绪定义
EMOTIONS = {
    "neutral": {
        "landmark_offset": [0, 0, 0],
        "pitch_range": (-0.05, 0.05),
        "description": ["面无表情，保持中性", "平静的面部表情", "自然放松的状态", "没有任何情绪变化"]
    },
    "happy": {
        "landmark_offset": [0, 0.05, 0],  # 嘴角上扬
        "pitch_range": (-0.1, 0.1),
        "description": ["开心的微笑", "嘴角上扬，露出开心的表情", "愉快的面部表情", "高兴的样子"]
    },
    "sad": {
        "landmark_offset": [0, -0.03, 0],  # 嘴角下垂
        "pitch_range": (-0.15, 0),
        "description": ["悲伤的表情", "嘴角下垂，看起来很失落", "难过的面部表情", "忧伤的样子"]
    },
    "angry": {
        "landmark_offset": [0.02, 0, 0],  # 眉头紧锁
        "pitch_range": (0, 0.15),
        "description": ["愤怒的表情", "眉头紧锁，嘴唇紧闭", "生气的面部表情", "发怒的样子"]
    },
    "surprised": {
        "landmark_offset": [0, 0.1, 0],  # 嘴巴张开
        "pitch_range": (-0.05, 0.1),
        "description": ["惊讶的表情", "眼睛睁大，嘴巴张开", "意外的面部表情", "吃惊的样子"]
    },
    "fearful": {
        "landmark_offset": [-0.01, 0, 0],
        "pitch_range": (-0.1, 0),
        "description": ["恐惧的表情", "害怕的面部表情", "紧张不安的样子", "畏惧的表情"]
    },
    "disgusted": {
        "landmark_offset": [0, -0.02, 0],
        "pitch_range": (0, 0.1),
        "description": ["厌恶的表情", "不喜欢的面部表情", "反感的表情", "恶心的表情"]
    },
}

EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprised": 4,
    "fearful": 5,
    "disgusted": 6,
}


def generate_base_landmarks():
    """生成基础面部关键点 (21个3D点)"""
    # LivePortrait 关键点分布
    base = np.array([
        [0.0, 0.0, 0.0],      # 0: 鼻尖
        [0.05, 0.02, 0.0],    # 1-2: 内眼角
        [-0.05, 0.02, 0.0],
        [0.0, 0.05, 0.01],    # 3-4: 外眼角
        [0.03, 0.06, 0.01],
        [-0.03, 0.06, 0.01],
        [0.08, 0.03, 0.0],    # 5-6: 眉头
        [-0.08, 0.03, 0.0],
        [0.1, 0.05, 0.01],    # 7-8: 眉梢
        [-0.1, 0.05, 0.01],
        [0.06, 0.08, 0.02],   # 9-10: 鼻翼
        [-0.06, 0.08, 0.02],
        [0.0, 0.1, 0.02],     # 11-12: 鼻基底
        [0.04, 0.12, 0.02],
        [-0.04, 0.12, 0.02],
        [0.0, 0.15, 0.01],    # 13: 上唇中心
        [0.02, 0.15, 0.01],   # 14-15: 嘴角
        [-0.02, 0.15, 0.01],
        [0.0, 0.18, 0.0],     # 16: 下唇中心
        [0.01, 0.2, 0.0],     # 17-20: 下巴和下颌
        [-0.01, 0.2, 0.0],
    ])
    return base


def apply_emotion(base_landmarks, emotion_name, intensity):
    """应用情绪偏移"""
    offset = np.array(EMOTIONS[emotion_name]["landmark_offset"])
    return base_landmarks + offset * intensity


def add_random_variation(landmarks, pose, noise_level=0.01):
    """添加随机变化"""
    landmarks = landmarks + np.random.randn(*landmarks.shape) * noise_level
    pose = pose + np.random.randn(len(pose)) * noise_level * 2
    return landmarks, pose


def generate_sample(
    emotion_name: str,
    intensity: float,
    text: str = None,
    variation: bool = True,
):
    """生成单个样本"""
    emotion_idx = EMOTION_MAP[emotion_name]

    # 获取文本
    if text is None:
        text = np.random.choice(EMOTIONS[emotion_name]["description"])

    # 生成关键点
    base = generate_base_landmarks()
    landmarks = apply_emotion(base, emotion_name, intensity)

    # 生成姿态
    pitch_range = EMOTIONS[emotion_name]["pitch_range"]
    pitch = np.random.uniform(*pitch_range) * intensity
    yaw = np.random.uniform(-0.1, 0.1) * intensity
    roll = np.random.uniform(-0.05, 0.05) * intensity
    t = np.random.randn(3) * 0.02
    scale = 1.0 + np.random.randn() * 0.05

    pose = [pitch, yaw, roll, t[0], t[1], scale]

    # 添加变化
    if variation:
        landmarks, pose = add_random_variation(landmarks, np.array(pose))

    return {
        "text": text,
        "landmarks": landmarks.tolist(),
        "pitch": float(pose[0]),
        "yaw": float(pose[1]),
        "roll": float(pose[2]),
        "t": [float(pose[3]), float(pose[4]), float(pose[5])],
        "scale": float(pose[5]),
        "emotion": emotion_idx,
        "intensity": float(intensity),
    }


def generate_dataset(
    output_path: str,
    samples_per_emotion: int = 100,
    intensity_range: tuple = (0.5, 1.0),
    balanced: bool = True,
):
    """
    生成完整数据集

    Args:
        output_path: 输出文件路径
        samples_per_emotion: 每种情绪的样本数
        intensity_range: 强度范围
        balanced: 是否平衡各情绪
    """
    samples = []

    emotions = list(EMOTIONS.keys())

    for emotion in emotions:
        for _ in range(samples_per_emotion):
            intensity = np.random.uniform(*intensity_range)
            sample = generate_sample(emotion, intensity)
            samples.append(sample)

    # 打乱
    np.random.shuffle(samples)

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"生成了 {len(samples)} 个样本到 {output_path}")

    # 统计
    emotion_counts = {}
    for s in samples:
        e = s["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    print("\n情绪分布:")
    for idx, name in EMOTION_MAP.items():
        count = emotion_counts.get(idx, 0)
        print(f"  {name:12s}: {count:4d} 样本")


def generate_split_dataset(
    output_dir: str,
    total_samples: int = 10000,
    val_ratio: float = 0.2,
):
    """生成训练/验证分割数据集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = int(total_samples * (1 - val_ratio))
    val_samples = total_samples - train_samples

    samples_per_emotion_train = train_samples // 7
    samples_per_emotion_val = val_samples // 7

    print("生成训练集...")
    generate_dataset(
        output_dir / "train.jsonl",
        samples_per_emotion=samples_per_emotion_train,
    )

    print("\n生成验证集...")
    generate_dataset(
        output_dir / "val.jsonl",
        samples_per_emotion=samples_per_emotion_val,
    )

    print(f"\n完成! 数据集保存在 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="生成 Text2Face 训练数据")

    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")

    parser.add_argument("--num-samples", type=int, default=1000,
                        help="每种情绪的样本数")

    parser.add_argument("--intensity-min", type=float, default=0.5,
                        help="最小强度")

    parser.add_argument("--intensity-max", type=float, default=1.0,
                        help="最大强度")

    parser.add_argument("--split", action="store_true",
                        help="生成训练/验证分割")

    parser.add_argument("--total", type=int, default=10000,
                        help="总样本数 (用于 --split)")

    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集比例")

    args = parser.parse_args()

    if args.split:
        generate_split_dataset(
            args.output,
            total_samples=args.total,
            val_ratio=args.val_ratio,
        )
    else:
        generate_dataset(
            args.output,
            samples_per_emotion=args.num_samples,
            intensity_range=(args.intensity_min, args.intensity_max),
        )


if __name__ == "__main__":
    main()
