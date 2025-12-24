"""
Text-to-Facial-Keypoints Model
文本到面部关键点生成模型

支持:
- 21个3D面部关键点 (LivePortrait格式)
- 6D头部姿态 (pitch, yaw, roll, tx, ty, scale)
- 时序平滑生成
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Dict, Optional
import math


class Text2FaceKeypoints(nn.Module):
    """
    文本到面部关键点生成模型

    输入: 文本描述
    输出:
        - landmarks: (B, T, 21, 3) 21个3D关键点
        - pose: (B, T, 6) 头部姿态 [pitch, yaw, roll, tx, ty, scale]
    """

    def __init__(
        self,
        text_encoder: str = "bert-base-chinese",
        hidden_dim: int = 768,
        num_landmarks: int = 21,
        num_pose_params: int = 6,
        latent_dim: int = 256,
        dropout: float = 0.1,
        use_temporal: bool = False,
        num_temporal_layers: int = 2,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.num_pose_params = num_pose_params
        self.use_temporal = use_temporal

        # 文本编码器
        self.text_encoder = AutoModel.from_pretrained(text_encoder)

        # 投影层: 文本特征 -> 潜在空间
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 表情嵌入 (可学习)
        self.expression_embedding = nn.Parameter(torch.randn(128))

        # 关键点解码器
        self.landmark_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_landmarks * 3),
        )

        # 姿态解码器
        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_pose_params),
        )

        # 时序平滑模块 (可选)
        if use_temporal:
            self.temporal_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim + 128,
                    nhead=8,
                    dim_feedforward=latent_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=num_temporal_layers,
            )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in [self.landmark_decoder, self.pose_decoder]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码文本

        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask

        Returns:
            text_feat: (B, D) 文本特征
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 使用 [CLS] token 或 mean pooling
        text_feat = outputs.last_hidden_state[:, 0, :]  # (B, D)

        return text_feat

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: (B, L) 或 (B, T, L) 文本 token ids
            attention_mask: (B, L) 或 (B, T, L) attention mask
            sequence_length: 生成的帧序列长度

        Returns:
            landmarks: (B, T, 21, 3) 面部关键点
            pose: (B, T, 6) 头部姿态
        """
        batch_size = input_ids.shape[0]

        # 处理输入形状
        if input_ids.dim() == 3:
            # 时序输入: (B, T, L)
            T, L = input_ids.shape[1], input_ids.shape[2]
            input_ids = input_ids.view(batch_size * T, L)
            attention_mask = attention_mask.view(batch_size * T, L)

        # 编码文本
        text_feat = self.encode_text(input_ids, attention_mask)

        # 投影到潜在空间
        text_feat = self.text_projection(text_feat)  # (B, D)

        # 添加表情嵌入
        B = text_feat.shape[0]
        expr_emb = self.expression_embedding.unsqueeze(0).expand(B, -1)
        feat = torch.cat([text_feat, expr_emb], dim=-1)  # (B, D + 128)

        # 时序处理
        if self.use_temporal and sequence_length > 1:
            # 扩展为序列
            feat = feat.unsqueeze(1).expand(-1, sequence_length, -1)
            feat = self.temporal_encoder(feat)  # (B, T, D + 128)
        else:
            # 单帧或无时序
            feat = feat.unsqueeze(1).expand(-1, sequence_length, -1)

        # 解码关键点和姿态
        landmarks = self.landmark_decoder(feat)  # (B, T, 21 * 3)
        pose = self.pose_decoder(feat)  # (B, T, 6)

        # 重塑关键点
        landmarks = landmarks.view(batch_size, sequence_length, self.num_landmarks, 3)

        return landmarks, pose

    def generate(
        self,
        text: str,
        tokenizer,
        sequence_length: int = 1,
        temperature: float = 1.0,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从文本生成关键点

        Args:
            text: 输入文本
            tokenizer: 文本分词器
            sequence_length: 生成序列长度
            temperature: 采样温度
            device: 设备

        Returns:
            landmarks: (T, 21, 3) 面部关键点
            pose: (T, 6) 头部姿态
        """
        self.eval()

        with torch.no_grad():
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # 生成
            landmarks, pose = self.forward(
                input_ids,
                attention_mask,
                sequence_length=sequence_length,
            )

            # 应用温度 (可选)
            if temperature != 1.0:
                landmarks = landmarks / temperature
                pose = pose / temperature

        return landmarks.squeeze(0).cpu(), pose.squeeze(0).cpu()


class Text2FaceKeypointsWithEmotion(Text2FaceKeypoints):
    """
    带情绪控制的文本到关键点模型

    额外输入:
        - emotion_label: 情绪标签
        - emotion_intensity: 情绪强度 [0, 1]
    """

    def __init__(
        self,
        num_emotions: int = 7,  # neutral, happy, sad, angry, surprised, fearful, disgusted
        emotion_embed_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)

        # 情绪嵌入
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embed_dim)

        # 强度调制
        self.intensity_gate = nn.Sequential(
            nn.Linear(1, emotion_embed_dim),
            nn.Sigmoid(),
        )

        # 更新投影层输入维度
        projected_dim = kwargs.get("hidden_dim", 768)
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + emotion_embed_dim, projected_dim),
            nn.LayerNorm(projected_dim),
            nn.ReLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
        emotion_intensity: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带情绪控制的前向传播

        Args:
            emotion_labels: (B,) 情绪标签索引
            emotion_intensity: (B,) 情绪强度 [0, 1]
        """
        batch_size = input_ids.shape[0]

        # 编码文本
        text_feat = self.encode_text(input_ids, attention_mask)

        # 处理情绪
        if emotion_labels is not None:
            emotion_feat = self.emotion_embedding(emotion_labels)

            if emotion_intensity is not None:
                # 强度调制
                gate = self.intensity_gate(emotion_intensity.unsqueeze(-1))
                emotion_feat = emotion_feat * gate

            # 融合文本和情绪特征
            text_feat = torch.cat([text_feat, emotion_feat], dim=-1)

        # 后续处理与基类相同
        text_feat = self.text_projection(text_feat)
        B = text_feat.shape[0]
        expr_emb = self.expression_embedding.unsqueeze(0).expand(B, -1)
        feat = torch.cat([text_feat, expr_emb], dim=-1)

        if self.use_temporal and sequence_length > 1:
            feat = feat.unsqueeze(1).expand(-1, sequence_length, -1)
            feat = self.temporal_encoder(feat)
        else:
            feat = feat.unsqueeze(1).expand(-1, sequence_length, -1)

        landmarks = self.landmark_decoder(feat)
        pose = self.pose_decoder(feat)
        landmarks = landmarks.view(batch_size, sequence_length, self.num_landmarks, 3)

        return landmarks, pose


# 情绪标签映射
EMOTION_LABELS = {
    0: "neutral",    # 中性
    1: "happy",      # 开心
    2: "sad",        # 悲伤
    3: "angry",      # 愤怒
    4: "surprised",  # 惊讶
    5: "fearful",    # 恐惧
    6: "disgusted",  # 厌恶
}

# 情绪描述增强
EMOTION_PROMPTS = {
    "neutral": [
        "面无表情，保持中性",
        "平静的面部表情",
        "没有任何情绪变化",
    ],
    "happy": [
        "开心地微笑",
        "嘴角上扬，露出开心的表情",
        "愉快的面部表情",
    ],
    "sad": [
        "悲伤的表情",
        "嘴角下垂，看起来很失落",
        "难过的面部表情",
    ],
    "angry": [
        "愤怒的表情",
        "眉头紧锁，嘴唇紧闭",
        "生气的面部表情",
    ],
    "surprised": [
        "惊讶的表情",
        "眼睛睁大，嘴巴张开",
        "意外的面部表情",
    ],
    "fearful": [
        "恐惧的表情",
        "害怕的面部表情",
        "紧张不安的样子",
    ],
    "disgusted": [
        "厌恶的表情",
        "不喜欢的面部表情",
        "反感的表情",
    ],
}


def get_model(
    model_type: str = "base",
    **kwargs
) -> nn.Module:
    """
    获取模型实例

    Args:
        model_type: "base" 或 "emotion"
        **kwargs: 模型参数

    Returns:
        model: Text2FaceKeypoints 实例
    """
    if model_type == "emotion":
        return Text2FaceKeypointsWithEmotion(**kwargs)
    else:
        return Text2FaceKeypoints(**kwargs)


if __name__ == "__main__":
    # 测试模型
    model = get_model(model_type="base")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试前向传播
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    attention_mask = torch.ones_like(input_ids)

    landmarks, pose = model(input_ids, attention_mask, sequence_length=seq_len)

    print(f"\nOutput shapes:")
    print(f"  Landmarks: {landmarks.shape}")  # (2, 10, 21, 3)
    print(f"  Pose: {pose.shape}")  # (2, 10, 6)
