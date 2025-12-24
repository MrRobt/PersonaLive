"""
关键点可视化工具

可视化生成的面部关键点，用于验证数据质量
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# LivePortrait 关键点连接关系
LANDMARK_CONNECTIONS = [
    # 轮廓
    (5, 7),   # 左眉
    (6, 8),   # 右眉
    (1, 3),   # 左眼
    (2, 4),   # 右眼
    (3, 5),   # 左眼-左眉
    (4, 6),   # 右眼-右眉
    (9, 11),  # 左鼻翼
    (10, 12), # 右鼻翼
    (13, 14), # 上唇
    (13, 15), # 上唇
    (16, 14), # 下唇
    (16, 15), # 下唇
    (17, 18), # 下巴
    (17, 19), # 下巴
    (17, 20), # 下巴
]

# 关键点颜色
EMOTION_COLORS = {
    0: "#808080",  # neutral - gray
    1: "#FFD700",  # happy - gold
    2: "#4169E1",  # sad - blue
    3: "#DC143C",  # angry - red
    4: "#FF69B4",  # surprised - pink
    5: "#9370DB",  # fearful - purple
    6: "#228B22",  # disgusted - green
}

EMOTION_NAMES = {
    0: "中性",
    1: "开心",
    2: "悲伤",
    3: "愤怒",
    4: "惊讶",
    5: "恐惧",
    6: "厌恶",
}


def load_samples(data_path, num_samples=None):
    """加载样本数据"""
    samples = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if num_samples and len(samples) >= num_samples:
                    break

    return samples


def visualize_landmarks(
    landmarks,
    pose,
    text,
    emotion,
    intensity,
    ax=None,
    show_connections=True,
    show_indices=False,
):
    """可视化单个样本的关键点"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    landmarks = np.array(landmarks)

    # 绘制连接线
    if show_connections:
        for i, j in LANDMARK_CONNECTIONS:
            if i < len(landmarks) and j < len(landmarks):
                ax.plot(
                    [landmarks[i, 0], landmarks[j, 0]],
                    [landmarks[i, 1], landmarks[j, 1]],
                    'k-', alpha=0.3, linewidth=1
                )

    # 绘制关键点
    color = EMOTION_COLORS.get(emotion, "#000000")
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c=color, s=50, zorder=3)

    # 绘制索引
    if show_indices:
        for i, (x, y) in enumerate(landmarks):
            ax.text(x, y + 0.01, str(i), ha='center', va='bottom', fontsize=8)

    # 设置标题
    emotion_name = EMOTION_NAMES.get(emotion, f"情绪{emotion}")
    title = f"{text}\n"
    title += f"{emotion_name} (强度: {intensity:.2f})\n"
    title += f"姿态: pitch={pose[0]:.3f}, yaw={pose[1]:.3f}, roll={pose[2]:.3f}"
    ax.set_title(title, fontsize=10)

    # 设置坐标轴
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.1, 0.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def visualize_samples_grid(
    samples,
    ncols=5,
    save_path=None,
):
    """可视化多个样本（网格布局）"""
    nrows = (len(samples) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten() if nrows > 1 else [axes]

    for i, sample in enumerate(samples):
        ax = axes[i]
        visualize_landmarks(
            landmarks=sample["landmarks"],
            pose=[
                sample["pitch"],
                sample["yaw"],
                sample["roll"],
                sample["t"][0],
                sample["t"][1],
                sample["scale"],
            ],
            text=sample["text"],
            emotion=sample.get("emotion", 0),
            intensity=sample.get("intensity", 0.5),
            ax=ax,
        )

    # 隐藏多余的子图
    for i in range(len(samples), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到 {save_path}")

    plt.show()


def visualize_emotion_distribution(samples, save_path=None):
    """可视化情绪分布"""
    emotions = [s.get("emotion", 0) for s in samples]
    emotion_counts = {}
    for e in emotions:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    fig, ax = plt.subplots(figsize=(10, 6))

    emotions_list = sorted(emotion_counts.keys())
    counts = [emotion_counts[e] for e in emotions_list]
    labels = [EMOTION_NAMES.get(e, f"情绪{e}") for e in emotions_list]
    colors = [EMOTION_COLORS.get(e, "#000000") for e in emotions_list]

    bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.7)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('情绪')
    ax.set_ylabel('样本数')
    ax.set_title('数据集情绪分布')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到 {save_path}")

    plt.show()


def visualize_intensity_distribution(samples, save_path=None):
    """可视化强度分布"""
    intensities = [s.get("intensity", 0.5) for s in samples]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(intensities, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('强度')
    ax.set_ylabel('样本数')
    ax.set_title('情绪强度分布')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到 {save_path}")

    plt.show()


def visualize_pose_statistics(samples, save_path=None):
    """可视化姿态统计"""
    pitches = [s["pitch"] for s in samples]
    yaws = [s["yaw"] for s in samples]
    rolls = [s["roll"] for s in samples]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(pitches, bins=30, edgecolor='black', alpha=0.7, color='red')
    axes[0].set_xlabel('Pitch (俯仰角)')
    axes[0].set_ylabel('样本数')
    axes[0].set_title('Pitch 分布')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(yaws, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Yaw (偏航角)')
    axes[1].set_ylabel('样本数')
    axes[1].set_title('Yaw 分布')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].hist(rolls, bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[2].set_xlabel('Roll (翻滚角)')
    axes[2].set_ylabel('样本数')
    axes[2].set_title('Roll 分布')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存到 {save_path}")

    plt.show()


def print_statistics(samples):
    """打印数据统计信息"""
    print("\n" + "=" * 50)
    print("数据集统计")
    print("=" * 50)

    print(f"总样本数: {len(samples)}")

    # 情绪分布
    emotions = [s.get("emotion", 0) for s in samples]
    emotion_counts = {}
    for e in emotions:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    print("\n情绪分布:")
    for e in sorted(emotion_counts.keys()):
        name = EMOTION_NAMES.get(e, f"情绪{e}")
        count = emotion_counts[e]
        ratio = count / len(samples) * 100
        print(f"  {name:12s}: {count:4d} ({ratio:5.1f}%)")

    # 强度统计
    intensities = [s.get("intensity", 0.5) for s in samples]
    print(f"\n强度统计:")
    print(f"  最小值: {min(intensities):.3f}")
    print(f"  最大值: {max(intensities):.3f}")
    print(f"  平均值: {np.mean(intensities):.3f}")
    print(f"  中位数: {np.median(intensities):.3f}")

    # 姿态统计
    pitches = [s["pitch"] for s in samples]
    yaws = [s["yaw"] for s in samples]
    rolls = [s["roll"] for s in samples]

    print(f"\n姿态统计:")
    print(f"  Pitch: {np.mean(pitches):.3f} ± {np.std(pitches):.3f}")
    print(f"  Yaw:   {np.mean(yaws):.3f} ± {np.std(yaws):.3f}")
    print(f"  Roll:  {np.mean(rolls):.3f} ± {np.std(rolls):.3f}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="可视化 Text2Face 数据集")

    parser.add_argument("--data", type=str, required=True,
                        help="数据文件路径 (.jsonl)")

    parser.add_argument("--num-samples", type=int, default=20,
                        help="可视化的样本数")

    parser.add_argument("--ncols", type=int, default=5,
                        help="网格列数")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出图片目录")

    parser.add_argument("--show-indices", action="store_true",
                        help="显示关键点索引")

    parser.add_argument("--stats-only", action="store_true",
                        help="只显示统计信息")

    args = parser.parse_args()

    # 加载样本
    samples = load_samples(args.data, args.num_samples)

    if args.stats_only:
        print_statistics(samples)
        return

    # 打印统计
    print_statistics(samples)

    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 可视化样本
    print("\n可视化样本...")
    save_path = f"{args.output_dir}/samples.png" if args.output_dir else None
    visualize_samples_grid(samples, ncols=args.ncols, save_path=save_path)

    # 可视化情绪分布
    print("可视化情绪分布...")
    save_path = f"{args.output_dir}/emotion_distribution.png" if args.output_dir else None
    visualize_emotion_distribution(samples, save_path=save_path)

    # 可视化强度分布
    print("可视化强度分布...")
    save_path = f"{args.output_dir}/intensity_distribution.png" if args.output_dir else None
    visualize_intensity_distribution(samples, save_path=save_path)

    # 可视化姿态统计
    print("可视化姿态统计...")
    save_path = f"{args.output_dir}/pose_statistics.png" if args.output_dir else None
    visualize_pose_statistics(samples, save_path=save_path)


if __name__ == "__main__":
    main()
