# Text2Face 数据集示例

本目录包含 Text2Face 项目的数据集示例文件。

## 文件说明

### 1. 训练数据格式 (sample_train.jsonl)

JSONL 格式，每行一个样本：

```json
{
  "text": "开心的微笑",
  "landmarks": [[x1,y1,z1], [x2,y2,z2], ..., [x21,y21,z21]],
  "pitch": 0.0,
  "yaw": 0.0,
  "roll": 0.0,
  "t": [tx, ty, tz],
  "scale": 1.0,
  "emotion": 1,
  "intensity": 0.8
}
```

**字段说明:**

| 字段 | 类型 | 说明 |
|------|------|------|
| text | string | 文本描述 |
| landmarks | list[[float]] | 21个3D关键点 (21, 3) |
| pitch | float | 俯仰角 (弧度) |
| yaw | float | 偏航角 (弧度) |
| roll | float | 翻滚角 (弧度) |
| t | list[float] | 平移向量 (3,) |
| scale | float | 缩放因子 |
| emotion | int | 情绪标签 (0-6) |
| intensity | float | 情绪强度 [0, 1] |

### 2. 标注文件格式 (annotations.json)

用于视频数据集的标注：

```json
{
  "dataset": "Dataset Name",
  "version": "1.0",
  "data": [
    {
      "video": "path/to/video.mp4",
      "text": "文本描述",
      "emotion": "happy",
      "intensity": 0.8
    }
  ]
}
```

## 情绪标签映射

| 标签值 | 情绪 | 英文 |
|--------|------|------|
| 0 | 中性 | neutral |
| 1 | 开心 | happy |
| 2 | 悲伤 | sad |
| 3 | 愤怒 | angry |
| 4 | 惊讶 | surprised |
| 5 | 恐惧 | fearful |
| 6 | 厌恶 | disgusted |

## 数据集准备方法

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

## 示例数据统计

| 文件 | 样本数 | 情绪分布 |
|------|--------|----------|
| sample_train.jsonl | 20 | 全部7种情绪 |
| sample_val.jsonl | 7 | 每种情绪1个 |

## 推荐的文本描述

### 情绪相关

```
中性:
- "面无表情"
- "平静的表情"
- "自然放松"

开心:
- "开心的微笑"
- "嘴角上扬"
- "愉快的表情"

悲伤:
- "悲伤的表情"
- "嘴角下垂"
- "难过的样子"

愤怒:
- "愤怒的表情"
- "眉头紧锁"
- "生气的样子"

惊讶:
- "惊讶的表情"
- "眼睛睁大"
- "意外的样子"

恐惧:
- "恐惧的表情"
- "害怕的样子"
- "紧张不安"

厌恶:
- "厌恶的表情"
- "不喜欢的表情"
- "反感的样子"
```

### 姿态相关

```
- "微微点头" (pitch > 0)
- "微微摇头" (pitch < 0)
- "向左看" (yaw > 0)
- "向右看" (yaw < 0)
- "歪头" (roll != 0)
```

## 关键点数据格式

LivePortrait 使用 21 个 3D 关键点：

```
索引 | 描述
-----|------
0    | 鼻尖
1-2  | 内眼角
3-4  | 外眼角
5-6  | 眉头
7-8  | 眉梢
9-10 | 鼻翼
11-12| 鼻基底
13   | 上唇中心
14-15| 嘴角
16   | 下唇中心
17   | 下巴中心
18-20| 下颌轮廓点
```

## 数据增强

预处理脚本会自动应用以下增强：

1. **旋转**: ±5度
2. **缩放**: 0.95-1.05倍
3. **平移**: ±0.01
4. **姿态抖动**: ±0.02

## 数据质量要求

1. **关键点精度**: 建议使用 LivePortrait 提取器
2. **文本描述**: 清晰、准确、与表情匹配
3. **情绪标注**: 一致、准确
4. **样本平衡**: 各情绪类别样本数尽量均衡

## 常见问题

### Q: 如何增加数据量？

A: 使用 `--augment` 参数启用数据增强，会自动为每个样本生成多种变体。

### Q: 如何处理多语言？

A: 修改 `configs/default.yaml` 中的 `text_encoder`：
- 中文: `bert-base-chinese`
- 英文: `bert-base-uncased`

### Q: 如何验证数据质量？

A: 使用示例脚本可视化关键点：

```bash
python examples/visualize_keypoints.py \
    --data data/processed/train.jsonl \
    --num-samples 10
```

## 外部数据集推荐

| 数据集 | 类型 | 链接 |
|--------|------|------|
| MEAD | 多情绪说话人头 | [GitHub](https://github.com/yzhou359/MEAD) |
| VoxCeleb2 | 说话人视频 | [官网](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) |
| CREMA-D | 情绪语音 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) |
| RAVDESS | 语音情绪 | [Kaggle](https://www.kaggle.com/datasets/shivamb/raided-speech-audio-emotion) |
