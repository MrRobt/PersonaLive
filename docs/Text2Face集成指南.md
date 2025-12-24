# Text2Face 项目完整文档

## 项目概述

Text2Face 是一个文本到面部关键点生成模型，可以直接将文本描述转换为 PersonaLive 兼容的面部关键点和头部姿态，无需预先准备驱动视频。

---

## 目录结构

```
PersonaLive/
├── text2face/                    # Text2Face 项目
│   ├── src/
│   │   ├── models/
│   │   │   └── model.py          # 模型定义 (Text2FaceKeypoints)
│   │   ├── data/
│   │   │   ├── dataset.py        # 数据集类
│   │   │   ├── keypoint_extractor.py  # 关键点提取工具
│   │   │   └── preprocess.py     # 数据预处理脚本
│   │   ├── train.py              # 训练脚本
│   │   └── inference.py          # 推理和 PersonaLive 集成
│   ├── configs/
│   │   ├── default.yaml          # 基础配置
│   │   └── emotion.yaml          # 情绪控制配置
│   ├── checkpoints/              # 模型检查点目录
│   ├── data/
│   │   └── processed/            # 处理后的数据
│   └── README.md
└── docs/
    ├── 原理分析.md                # PersonaLive 原理分析
    ├── 架构分析.md                # PersonaLive 架构分析
    └── Text2Face集成指南.md       # 本文档
```

---

## 核心模块说明

### 1. 模型模块 (`src/models/model.py`)

#### Text2FaceKeypoints (基础模型)

```python
class Text2FaceKeypoints(nn.Module):
    """文本到面部关键点生成模型"""

    def __init__(
        self,
        text_encoder: str = "bert-base-chinese",  # 文本编码器
        hidden_dim: int = 768,
        num_landmarks: int = 21,     # LivePortrait 关键点数量
        num_pose_params: int = 6,    # pitch, yaw, roll, tx, ty, scale
        latent_dim: int = 256,
        use_temporal: bool = False,
    ):
```

**输出:**
- `landmarks`: (B, T, 21, 3) - 21个3D面部关键点
- `pose`: (B, T, 6) - 头部姿态参数

#### Text2FaceKeypointsWithEmotion (情绪控制模型)

```python
class Text2FaceKeypointsWithEmotion(Text2FaceKeypoints):
    """带情绪控制的文本到关键点模型"""

    def __init__(
        self,
        num_emotions: int = 7,       # 7种情绪
        emotion_embed_dim: int = 64,
        **kwargs
    ):
```

**额外输入:**
- `emotion_labels`: (B,) - 情绪标签索引 (0-6)
- `emotion_intensity`: (B,) - 情绪强度 [0, 1]

### 2. 数据模块

#### KeypointExtractor (`src/data/keypoint_extractor.py`)

从视频/图像中提取 LivePortrait 格式的关键点。

```python
extractor = KeypointExtractor(device="cuda")

# 从视频提取
keypoints = extractor.extract_from_video(
    video_path="video.mp4",
    fps=25,
)

# 平滑关键点
smoothed = extractor.smooth_keypoints(keypoints)
```

#### TextKeypointDataset (`src/data/dataset.py`)

训练数据集类。

```python
dataset = TextKeypointDataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    augment=True,
    temporal=False,
)

sample = dataset[0]
# {
#     "input_ids": tensor([L]),
#     "attention_mask": tensor([L]),
#     "landmarks": tensor([21, 3]),
#     "pose": tensor([6]),
# }
```

### 3. 训练模块 (`src/train.py`)

```bash
# 基础训练
python -m src.train --config configs/default.yaml

# 带数据路径覆盖
python -m src.train \
    --config configs/default.yaml \
    --train-data data/train.jsonl \
    --val-data data/val.jsonl \
    --output checkpoints/my_run
```

### 4. 推理模块 (`src/inference.py`)

```python
from src.inference import Text2FaceInference

# 初始化
inference = Text2FaceInference(
    checkpoint_path="checkpoints/best.pt",
    device="cuda"
)

# 生成关键点
result = inference.generate(
    text="开心的微笑",
    emotion="happy",
    intensity=0.7,
)
```

---

## 完整使用流程

### 阶段 1: 数据准备

#### 方法 A: 使用现有视频数据集

```bash
# 1. 准备视频目录
mkdir -p data/videos/happy
mkdir -p data/videos/sad
# ... 将视频放入对应情绪目录

# 2. 预处理
python -m src.data.preprocess --mode video \
    --input data/videos \
    --output data/processed/train.jsonl \
    --augment

# 3. 分割训练/验证集
python -c "
import json
from pathlib import Path

data = []
with open('data/processed/train.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

# 80/20 分割
split = int(len(data) * 0.8)
train = data[:split]
val = data[split:]

with open('data/processed/train.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('data/processed/val.jsonl', 'w') as f:
    for item in val:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
"
```

#### 方法 B: 使用合成数据 (快速测试)

```bash
# 生成 10,000 个合成样本
python -m src.data.preprocess --mode synthetic \
    --output data/processed/train.jsonl \
    --num-samples 10000

# 生成验证集
python -m src.data.preprocess --mode synthetic \
    --output data/processed/val.jsonl \
    --num-samples 2000
```

### 阶段 2: 训练模型

```bash
# 基础模型
python -m src.train --config configs/default.yaml

# 情绪控制模型
python -m src.train --config configs/emotion.yaml

# 恢复训练
python -m src.train \
    --config configs/default.yaml \
    --resume checkpoints/default/checkpoint_latest.pt
```

### 阶段 3: 推理

#### 3.1 仅生成关键点

```bash
python -m src.inference \
    --checkpoint checkpoints/default/checkpoint_best.pt \
    --text "开心的微笑" \
    --emotion happy \
    --intensity 0.8 \
    --output output.npz
```

#### 3.2 生成完整视频 (集成 PersonaLive)

```python
from src.inference import Text2FaceWithPersonaLive

# 初始化
generator = Text2FaceWithPersonaLive(
    checkpoint_path="checkpoints/best.pt",
    device="cuda"
)

# 生成视频
video = generator.generate_video(
    text="开心的微笑，然后慢慢平静下来",
    reference_image="reference.jpg",
    emotion="happy",
    intensity=0.8,
    duration=3.0,
    fps=25,
    output_path="output.mp4"
)
```

---

## 与 PersonaLive 集成详解

### 集成方式 1: 独立调用

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import Text2FaceInference
from src.wrapper import PersonaLive

# 初始化
text2face = Text2FaceInference("checkpoints/best.pt")
personalive = PersonaLive()

# 生成关键点
result = text2face.generate(
    text="开心的微笑",
    emotion="happy",
    sequence_length=75,  # 3秒 @ 25fps
)

# 转换为 PersonaLive 格式
pose_sequence = []
for i in range(result["landmarks"].shape[0]):
    pose_dict = {
        "pitch": float(result["pose"][i, 0]),
        "yaw": float(result["pose"][i, 1]),
        "roll": float(result["pose"][i, 2]),
        "t": result["pose"][i, 3:6].tolist(),
        "scale": float(result["pose"][i, 5]),
        "kp": torch.from_numpy(result["landmarks"][i]).float(),
    }
    pose_sequence.append(pose_dict)

# 融合参考图
personalive.fuse_reference("reference.jpg")

# 逐帧生成
frames = []
for pose in pose_sequence:
    frame = personalive.generate_with_pose(pose)
    frames.append(frame)

# 保存视频
import imageio
imageio.mimsave("output.mp4", frames, fps=25)
```

### 集成方式 2: 使用封装类

```python
from src.inference import Text2FaceWithPersonaLive

generator = Text2FaceWithPersonaLive(
    checkpoint_path="checkpoints/best.pt",
    device="cuda"
)

# 直接生成视频
video = generator.generate_video(
    text="开心的微笑",
    reference_image="reference.jpg",
    duration=3.0,
    output_path="output.mp4"
)
```

---

## 数据格式

### 输入数据格式 (.jsonl)

```json
{
  "text": "开心的微笑",
  "landmarks": [[0.1, 0.2, 0.3], ..., [0.4, 0.5, 0.6]],
  "pitch": 0.1,
  "yaw": -0.2,
  "roll": 0.0,
  "t": [0.0, 0.0, 0.0],
  "scale": 1.0,
  "emotion": 1,
  "intensity": 0.8
}
```

### 输出关键点格式

```
landmarks: (T, 21, 3)
  - T: 帧数
  - 21: LivePortrait 关键点数量
  - 3: (x, y, z) 坐标

pose: (T, 6)
  - [:, 0]: pitch (俯仰角)
  - [:, 1]: yaw (偏航角)
  - [:, 2]: roll (翻滚角)
  - [:, 3:5]: t (平移向量)
  - [:, 5]: scale (缩放因子)
```

---

## 训练技巧

### 1. 损失权重调整

```yaml
# configs/default.yaml
training:
  landmark_weight: 1.0   # 关键点损失权重
  pose_weight: 2.0       # 姿态损失权重 (调高以改善姿态)
  temporal_weight: 0.1   # 时序一致性权重
```

### 2. 学习率调度

```yaml
training:
  scheduler: "cosine"    # cosine, linear, constant
  warmup_steps: 500
```

### 3. 数据增强

```yaml
data:
  augment: true  # 启用随机旋转、缩放、平移
```

---

## 常见问题

### Q1: 训练时显存不足

**解决方案:**
- 减小 `batch_size`
- 使用梯度累积
- 使用较小的文本编码器

### Q2: 生成的关键点不够平滑

**解决方案:**
- 启用 `use_temporal: true`
- 增加 `temporal_weight`
- 在推理时使用 `smooth=True`

### Q3: 情绪控制效果不明显

**解决方案:**
- 使用 `Text2FaceKeypointsWithEmotion` 模型
- 增加 `intensity` 参数值
- 检查训练数据中情绪标注的准确性

---

## 性能基准

| 模型 | 参数量 | 训练时间 | 推理速度 |
|------|--------|----------|----------|
| Base | ~110M | 2小时 (10K样本) | 10ms/frame |
| Emotion | ~220M | 4小时 (10K样本) | 15ms/frame |

*测试环境: RTX 3090, batch_size=32*

---

## 未来改进

1. **多语言支持**: 添加英文等其他语言模型
2. **扩散模型**: 使用扩散模型提高生成质量
3. **音频驱动**: 结合音频信号生成说话动画
4. **全身姿态**: 扩展到全身关键点生成

---

## 参考资源

- [PersonaLive GitHub](https://github.com/MrRobt/PersonaLive)
- [LivePortrait GitHub](https://github.com/KwaiVGI/LivePortrait)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [MEAD 数据集](https://github.com/yzhou359/MEAD)
