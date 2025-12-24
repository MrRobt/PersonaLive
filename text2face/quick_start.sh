#!/bin/bash
# Text2Face 快速启动脚本

set -e

echo "======================================"
echo "  Text2Face 快速启动脚本"
echo "======================================"

# 1. 创建虚拟环境
echo ""
echo "[1/5] 创建虚拟环境..."
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
echo ""
echo "[2/5] 安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. 生成合成数据
echo ""
echo "[3/5] 生成合成训练数据..."
python -m src.data.preprocess --mode synthetic \
    --output data/processed/train.jsonl \
    --num-samples 5000

python -m src.data.preprocess --mode synthetic \
    --output data/processed/val.jsonl \
    --num-samples 1000

# 4. 训练模型
echo ""
echo "[4/5] 训练模型..."
python -m src.train --config configs/default.yaml

# 5. 测试推理
echo ""
echo "[5/5] 测试推理..."
python -m src.inference \
    --checkpoint checkpoints/default/checkpoint_best.pt \
    --text "开心的微笑" \
    --emotion happy \
    --output output.npz

echo ""
echo "======================================"
echo "  完成!"
echo "======================================"
echo ""
echo "模型已保存到: checkpoints/default/"
echo "输出已保存到: output.npz"
echo ""
echo "下一步:"
echo "  1. 准备真实数据集进行训练"
echo "  2. 调整 configs/default.yaml 中的参数"
echo "  3. 运行 python -m src.train --config configs/default.yaml"
echo ""
