"""
Text2Face 基础使用示例

演示如何使用 Text2Face 模型从文本生成面部关键点
"""

import torch
import numpy as np
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model import get_model, EMOTION_LABELS
from transformers import AutoTokenizer


def example_1_basic_generation():
    """示例 1: 基础关键点生成"""
    print("=" * 50)
    print("示例 1: 基础关键点生成")
    print("=" * 50)

    # 创建模型
    model = get_model(model_type="base")
    model.eval()

    # 创建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 准备输入
    text = "开心的微笑"
    inputs = tokenizer(text, return_tensors="pt")

    # 生成
    with torch.no_grad():
        landmarks, pose = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    print(f"\n文本: {text}")
    print(f"关键点形状: {landmarks.shape}")  # (1, 1, 21, 3)
    print(f"姿态形状: {pose.shape}")  # (1, 1, 6)
    print(f"\n关键点 (前5个):\n{landmarks[0, 0, :5]}")
    print(f"\n姿态: pitch={pose[0, 0, 0]:.4f}, yaw={pose[0, 0, 1]:.4f}, roll={pose[0, 0, 2]:.4f}")


def example_2_emotion_control():
    """示例 2: 情绪控制生成"""
    print("\n" + "=" * 50)
    print("示例 2: 情绪控制生成")
    print("=" * 50)

    # 创建情绪控制模型
    model = get_model(
        model_type="emotion",
        num_emotions=len(EMOTION_LABELS),
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 测试不同情绪
    text = "说话"
    emotions = [0, 1, 2, 3, 4]  # neutral, happy, sad, angry, surprised

    print(f"\n基础文本: {text}")
    print("\n不同情绪下的生成:")

    for emotion_idx in emotions:
        inputs = tokenizer(text, return_tensors="pt")
        emotion = torch.tensor([emotion_idx])
        intensity = torch.tensor([0.7])

        with torch.no_grad():
            landmarks, pose = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                emotion_labels=emotion,
                emotion_intensity=intensity,
            )

        emotion_name = EMOTION_LABELS[emotion_idx]
        print(f"\n  {emotion_name:12s}: pose=({pose[0, 0, 0]:.2f}, {pose[0, 0, 1]:.2f}, {pose[0, 0, 2]:.2f})")


def example_3_sequence_generation():
    """示例 3: 时序生成"""
    print("\n" + "=" * 50)
    print("示例 3: 时序生成")
    print("=" * 50)

    model = get_model(
        model_type="base",
        use_temporal=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 生成序列
    text = "开心的表情"
    sequence_length = 10
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        landmarks, pose = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            sequence_length=sequence_length,
        )

    print(f"\n文本: {text}")
    print(f"序列长度: {sequence_length}")
    print(f"关键点序列形状: {landmarks.shape}")  # (1, 10, 21, 3)
    print(f"姿态序列形状: {pose.shape}")  # (1, 10, 6)

    # 打印前3帧的姿态
    print(f"\n前3帧的姿态:")
    for i in range(min(3, sequence_length)):
        print(f"  帧 {i+1}: pitch={pose[0, i, 0]:.4f}, yaw={pose[0, i, 1]:.4f}, roll={pose[0, i, 2]:.4f}")


def example_4_save_and_load():
    """示例 4: 保存和加载"""
    print("\n" + "=" * 50)
    print("示例 4: 保存和加载关键点")
    print("=" * 50)

    # 创建模型和生成
    model = get_model()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    text = "惊讶的表情"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        landmarks, pose = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # 保存到文件
    output_path = Path("example_output.npz")
    np.savez(
        output_path,
        landmarks=landmarks.numpy(),
        pose=pose.numpy(),
        text=text,
    )

    print(f"\n已保存到: {output_path}")

    # 加载
    data = np.load(output_path)
    print(f"加载的关键点形状: {data['landmarks'].shape}")
    print(f"加载的姿态形状: {data['pose'].shape}")
    print(f"文本: {data['text']}")


def example_5_text_prompts():
    """示例 5: 常用文本提示"""
    print("\n" + "=" * 50)
    print("示例 5: 常用文本提示")
    print("=" * 50)

    # 常用提示
    prompts = {
        "中性": ["面无表情", "平静的表情", "自然放松"],
        "开心": ["开心的微笑", "嘴角上扬", "愉快的表情"],
        "悲伤": ["悲伤的表情", "嘴角下垂", "难过的样子"],
        "惊讶": ["惊讶的表情", "眼睛睁大", "意外的样子"],
    }

    model = get_model()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    print("\n推荐的文本提示:")

    for emotion, texts in prompts.items():
        print(f"\n{emotion}:")
        for t in texts:
            print(f"  - '{t}'")

    # 测试一个提示
    test_prompt = prompts["开心"][0]
    inputs = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        landmarks, pose = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    print(f"\n测试 '{test_prompt}':")
    print(f"  生成的关键点形状: {landmarks.shape}")
    print(f"  生成的姿态: {pose[0, 0].numpy()}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("  Text2Face 使用示例")
    print("=" * 50)

    example_1_basic_generation()
    example_2_emotion_control()
    example_3_sequence_generation()
    example_4_save_and_load()
    example_5_text_prompts()

    print("\n" + "=" * 50)
    print("  所有示例运行完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
