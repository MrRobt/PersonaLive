# Text2Face - 文本到面部关键点生成

将文本描述转换为 PersonaLive 兼容的面部关键点和头部姿态。

## 特性

- 从文本直接生成 21 个 3D 面部关键点
- 生成 6D 头部姿态 (pitch, yaw, roll, tx, ty, scale)
- 支持情绪控制和强度调节
- 与 PersonaLive 无缝集成
- 支持时序平滑生成

## 安装

```bash
pip install torch transformers
pip install opencv-python numpy tqdm pyyaml
```

## 快速开始

### 1. 准备数据

```bash
# 使用合成数据测试
python -m src.data.preprocess --mode synthetic \
    --input dummy \
    --output data/processed/train.jsonl \
    --num-samples 1000
```

### 2. 训练模型

```bash
python -m src.train --config configs/default.yaml
```

### 3. 推理

```bash
# 生成关键点
python -m src.inference --checkpoint checkpoints/default/checkpoint_best.pt \
    --text "开心的微笑" \
    --emotion happy \
    --output output.npz

# 生成视频 (需要 PersonaLive)
python -m src.inference --checkpoint checkpoints/default/checkpoint_best.pt \
    --text "开心的微笑" \
    --reference reference.jpg \
    --output output.mp4
```

## 项目结构

```
text2face/
├── src/
│   ├── models/
│   │   └── model.py          # 模型定义
│   ├── data/
│   │   ├── dataset.py        # 数据集类
│   │   ├── keypoint_extractor.py  # 关键点提取
│   │   └── preprocess.py     # 数据预处理
│   ├── train.py              # 训练脚本
│   └── inference.py          # 推理脚本
├── configs/
│   ├── default.yaml          # 基础配置
│   └── emotion.yaml          # 情绪控制配置
├── checkpoints/              # 模型检查点
└── data/
    └── processed/            # 处理后的数据
```

## 模型架构

```
文本输入 (BERT/ RoBERTa)
    │
    ▼
文本编码器 (预训练)
    │
    ▼
投影层 + 表情嵌入
    │
    ├─► 关键点解码器 ──► 21 个 3D 关键点
    │
    └─► 姿态解码器 ──► 6D 头部姿态
```

## 情绪控制

支持 7 种情绪:

| 标签 | 情绪 |
|------|------|
| 0 | neutral (中性) |
| 1 | happy (开心) |
| 2 | sad (悲伤) |
| 3 | angry (愤怒) |
| 4 | surprised (惊讶) |
| 5 | fearful (恐惧) |
| 6 | disgusted (厌恶) |

## 数据集准备

### 方法 1: 从视频提取

```bash
python -m src.data.preprocess --mode video \
    --input /path/to/videos \
    --output data/processed/train.jsonl \
    --augment
```

### 方法 2: 使用标注文件

```bash
python -m src.data.preprocess --mode labeled \
    --input annotations.json \
    --output data/processed/train.jsonl
```

### 方法 3: 合成数据

```bash
python -m src.data.preprocess --mode synthetic \
    --output data/processed/train.jsonl \
    --num-samples 10000
```

## 与 PersonaLive 集成

```python
from src.inference import Text2FaceWithPersonaLive

# 初始化
generator = Text2FaceWithPersonaLive(
    checkpoint_path="checkpoints/best.pt",
    device="cuda"
)

# 生成视频
video = generator.generate_video(
    text="开心的微笑",
    reference_image="reference.jpg",
    emotion="happy",
    intensity=0.8,
    duration=3.0,
    output_path="output.mp4"
)
```

## 代码示例

### 基础推理

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

print(f"Landmarks: {result['landmarks'].shape}")  # (21, 3)
print(f"Pose: {result['pose'].shape}")  # (6,)
```

### 生成序列

```python
# 生成多帧
prompts = ["开心的表情", "然后慢慢平静下来", "最后保持中性"]

result = inference.generate_sequence(
    text_prompts=prompts,
    fps=25,
    smooth=True
)

print(f"Sequence: {result['landmarks'].shape}")  # (T, 21, 3)
```

## 配置说明

### 模型配置

- `type`: 模型类型 ("base" 或 "emotion")
- `text_encoder`: 预训练文本编码器
- `hidden_dim`: 隐藏层维度
- `latent_dim`: 潜在空间维度
- `use_temporal`: 是否使用时序编码

### 训练配置

- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `landmark_weight`: 关键点损失权重
- `pose_weight`: 姿态损失权重

## 性能优化

1. **使用更大的文本编码器**: `hfl/chinese-roberta-wwm-ext-large`
2. **启用时序模式**: 设置 `use_temporal: true`
3. **增加数据增强**: 设置 `augment: true`
4. **调整损失权重**: 根据验证集调整 `landmark_weight` 和 `pose_weight`

## 限制

- 仅支持面部和头部关键点
- 训练数据质量影响生成效果
- 需要与 PersonaLive 配合使用以生成完整视频

## 许可

Apache License 2.0

## 参考

- [PersonaLive](https://github.com/MrRobt/PersonaLive)
- [LivePortrait](https://github.com/KwaiVGI/LivePortrait)
- [Transformers](https://github.com/huggingface/transformers)
