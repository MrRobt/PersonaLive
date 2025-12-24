"""
数据预处理脚本

从视频/音频数据集构建训练数据:
1. 提取关键点
2. 转录音频为文本
3. 构建文本-关键点配对数据
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

try:
    from keypoint_extractor import KeypointExtractor, save_keypoints
except ImportError:
    from src.data.keypoint_extractor import KeypointExtractor, save_keypoints


# 情绪标签映射
EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprised": 4,
    "fearful": 5,
    "disgusted": 6,
}

# 情绪描述增强
EMOTION_DESCRIPTIONS = {
    0: [
        "面无表情，保持中性",
        "平静的面部表情",
        "没有任何情绪变化",
        "自然放松的状态",
    ],
    1: [
        "开心地微笑",
        "嘴角上扬，露出开心的表情",
        "愉快的面部表情",
        "高兴的样子",
    ],
    2: [
        "悲伤的表情",
        "嘴角下垂，看起来很失落",
        "难过的面部表情",
        "忧伤的样子",
    ],
    3: [
        "愤怒的表情",
        "眉头紧锁，嘴唇紧闭",
        "生气的面部表情",
        "发怒的样子",
    ],
    4: [
        "惊讶的表情",
        "眼睛睁大，嘴巴张开",
        "意外的面部表情",
        "吃惊的样子",
    ],
    5: [
        "恐惧的表情",
        "害怕的面部表情",
        "紧张不安的样子",
        "畏惧的表情",
    ],
    6: [
        "厌恶的表情",
        "不喜欢的面部表情",
        "反感的表情",
        "恶心的样子",
    ],
}


def transcribe_audio(
    audio_path: str,
    model: str = "base",
    language: str = "zh",
) -> str:
    """
    使用 Whisper 转录音频

    Args:
        audio_path: 音频文件路径
        model: Whisper 模型大小
        language: 语言代码

    Returns:
        转录文本
    """
    try:
        import whisper
        whisper_model = whisper.load_model(model)
        result = whisper_model.transcribe(audio_path, language=language)
        return result["text"]
    except ImportError:
        print("Warning: whisper not installed, using fallback")
        return ""


def extract_keypoints_from_video(
    video_path: str,
    output_dir: str,
    fps: int = 25,
    smooth: bool = True,
) -> List[Dict]:
    """
    从视频提取关键点

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        fps: 采样帧率
        smooth: 是否平滑

    Returns:
        关键点列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取关键点
    extractor = KeypointExtractor()
    keypoints = extractor.extract_from_video(video_path, fps=fps)

    # 平滑
    if smooth and len(keypoints) > 1:
        keypoints = extractor.smooth_keypoints(keypoints)

    # 保存
    video_name = Path(video_path).stem
    save_keypoints(keypoints, output_dir / f"{video_name}.npz")

    return keypoints


def create_training_samples(
    video_path: str,
    text: str,
    emotion: int = 0,
    intensity: float = 0.5,
    keypoints: List[Dict] = None,
) -> List[Dict]:
    """
    创建训练样本

    Args:
        video_path: 视频路径
        text: 文本描述
        emotion: 情绪标签
        intensity: 情绪强度
        keypoints: 预提取的关键点 (可选)

    Returns:
        训练样本列表
    """
    samples = []

    # 如果没有提供关键点，提取
    if keypoints is None:
        extractor = KeypointExtractor()
        keypoints = extractor.extract_from_video(video_path)

    # 为每一帧创建样本
    for kp in keypoints:
        sample = {
            "text": text,
            "landmarks": kp["landmarks"].tolist(),
            "pitch": kp["pitch"],
            "yaw": kp["yaw"],
            "roll": kp["roll"],
            "t": kp["t"].tolist(),
            "scale": kp["scale"],
            "emotion": emotion,
            "intensity": intensity,
        }
        samples.append(sample)

    return samples


def create_augmented_samples(
    base_text: str,
    emotion: int,
    keypoints: List[Dict],
) -> List[Dict]:
    """
    创建增强样本 (使用不同的文本描述)

    Args:
        base_text: 基础文本
        emotion: 情绪标签
        keypoints: 关键点序列

    Returns:
        增强样本列表
    """
    samples = []
    descriptions = EMOTION_DESCRIPTIONS.get(emotion, [base_text])

    # 为每种描述创建样本
    for desc in descriptions:
        for kp in keypoints:
            sample = {
                "text": desc,
                "landmarks": kp["landmarks"].tolist(),
                "pitch": kp["pitch"],
                "yaw": kp["yaw"],
                "roll": kp["roll"],
                "t": kp["t"].tolist(),
                "scale": kp["scale"],
                "emotion": emotion,
                "intensity": 0.7,  # 默认强度
            }
            samples.append(sample)

    return samples


def process_video_dataset(
    video_dir: str,
    output_path: str,
    use_whisper: bool = False,
    augment: bool = True,
) -> None:
    """
    处理视频数据集

    Args:
        video_dir: 视频目录
        output_path: 输出文件路径 (.jsonl)
        use_whisper: 是否使用 Whisper 转录
        augment: 是否数据增强
    """
    video_dir = Path(video_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # 查找所有视频
    video_files = list(video_dir.rglob("*.mp4")) + list(video_dir.rglob("*.avi"))

    print(f"Found {len(video_files)} videos")

    for video_path in tqdm(video_files, desc="Processing videos"):
        # 尝试从路径推断情绪
        emotion = 0  # 默认中性
        for emo_name, emo_idx in EMOTION_MAP.items():
            if emo_name in video_path.parent.name.lower() or emo_name in video_path.stem.lower():
                emotion = emo_idx
                break

        # 获取文本
        if use_whisper:
            # 转录音频
            text = transcribe_audio(str(video_path))
            if not text:
                text = EMOTION_DESCRIPTIONS[emotion][0]
        else:
            # 使用情绪描述
            text = EMOTION_DESCRIPTIONS[emotion][0]

        # 创建样本
        samples = create_training_samples(
            str(video_path),
            text=text,
            emotion=emotion,
        )

        # 数据增强
        if augment:
            aug_samples = create_augmented_samples(
                text,
                emotion,
                [s for s in samples if "landmarks" in s],
            )
            samples.extend(aug_samples)

        all_samples.extend(samples)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_samples)} samples to {output_path}")


def process_labeled_dataset(
    annotation_file: str,
    output_path: str,
    video_dir: str = None,
) -> None:
    """
    处理带标注的数据集

    Args:
        annotation_file: 标注文件 (JSON/CSV)
        output_path: 输出文件路径
        video_dir: 视频目录 (可选)
    """
    annotation_file = Path(annotation_file)

    # 读取标注
    with open(annotation_file, "r", encoding="utf-8") as f:
        if annotation_file.suffix == ".json":
            annotations = json.load(f)
        else:
            # CSV 格式
            import csv
            reader = csv.DictReader(f)
            annotations = list(reader)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for anno in tqdm(annotations, desc="Processing annotations"):
        # 获取视频路径
        if video_dir:
            video_path = Path(video_dir) / anno["video"]
        else:
            video_path = Path(anno["video"])

        if not video_path.exists():
            print(f"Warning: {video_path} not found, skipping")
            continue

        # 获取情绪
        emotion = EMOTION_MAP.get(anno.get("emotion", "neutral"), 0)
        intensity = float(anno.get("intensity", 0.5))

        # 创建样本
        samples = create_training_samples(
            str(video_path),
            text=anno.get("text", EMOTION_DESCRIPTIONS[emotion][0]),
            emotion=emotion,
            intensity=intensity,
        )

        all_samples.extend(samples)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_samples)} samples to {output_path}")


def create_synthetic_data(
    output_path: str,
    num_samples: int = 1000,
) -> None:
    """
    创建合成训练数据 (用于测试)

    Args:
        output_path: 输出文件路径
        num_samples: 样本数量
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []

    for _ in range(num_samples):
        # 随机情绪
        emotion = np.random.randint(0, 7)

        # 随机强度
        intensity = np.random.uniform(0.3, 1.0)

        # 随机文本
        text = np.random.choice(EMOTION_DESCRIPTIONS[emotion])

        # 随机关键点 (基于情绪)
        base_landmarks = np.random.randn(21, 3) * 0.1

        # 情绪偏移
        emotion_offset = {
            0: [0, 0, 0],      # neutral
            1: [0, 0.05, 0],   # happy (嘴角上扬)
            2: [0, -0.03, 0],  # sad (嘴角下垂)
            3: [0.02, 0, 0],   # angry (眉头紧锁)
            4: [0, 0.1, 0],    # surprised (嘴巴张开)
            5: [-0.01, 0, 0],  # fearful
            6: [0, -0.02, 0],  # disgusted
        }

        offset = np.array(emotion_offset[emotion]) * intensity
        landmarks = base_landmarks + offset

        # 随机姿态
        pitch = np.random.uniform(-0.3, 0.3) * intensity
        yaw = np.random.uniform(-0.5, 0.5) * intensity
        roll = np.random.uniform(-0.2, 0.2) * intensity
        t = np.random.randn(3) * 0.02
        scale = 1.0 + np.random.randn() * 0.05

        sample = {
            "text": text,
            "landmarks": landmarks.tolist(),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "t": t.tolist(),
            "scale": float(scale),
            "emotion": emotion,
            "intensity": float(intensity),
        }

        samples.append(sample)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Created {num_samples} synthetic samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Data preprocessing for Text2Face")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["video", "labeled", "synthetic"],
                        help="Processing mode")

    parser.add_argument("--input", type=str, required=True,
                        help="Input directory/file")

    parser.add_argument("--output", type=str, required=True,
                        help="Output file path (.jsonl)")

    parser.add_argument("--whisper", action="store_true",
                        help="Use Whisper for transcription")

    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation")

    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of synthetic samples")

    args = parser.parse_args()

    if args.mode == "video":
        process_video_dataset(
            video_dir=args.input,
            output_path=args.output,
            use_whisper=args.whisper,
            augment=args.augment,
        )
    elif args.mode == "labeled":
        process_labeled_dataset(
            annotation_file=args.input,
            output_path=args.output,
        )
    elif args.mode == "synthetic":
        create_synthetic_data(
            output_path=args.output,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
