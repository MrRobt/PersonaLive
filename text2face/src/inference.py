"""
推理代码

Text2FaceKeypoints 模型推理，集成 PersonaLive
"""

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer

from src.models.model import get_model, EMOTION_PROMPTS


class Text2FaceInference:
    """
    文本到面部关键点推理
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = "cuda",
    ):
        """
        初始化推理器

        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
            device: 运行设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # 加载配置
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        # 创建模型
        model_config = config.get("model", {})
        self.model_type = model_config.get("type", "base")

        self.model = get_model(
            model_type=self.model_type,
            **model_config
        )

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 创建 tokenizer
        text_encoder = model_config.get("text_encoder", "bert-base-chinese")
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)

        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def generate(
        self,
        text: str,
        emotion: str = None,
        intensity: float = 0.7,
        sequence_length: int = 1,
        temperature: float = 1.0,
    ) -> dict:
        """
        从文本生成关键点

        Args:
            text: 输入文本
            emotion: 情绪标签 (可选)
            intensity: 情绪强度 [0, 1]
            sequence_length: 生成序列长度
            temperature: 采样温度

        Returns:
            dict containing:
                - landmarks: (T, 21, 3) 关键点
                - pose: (T, 6) 姿态
        """
        # 增强文本
        if emotion and emotion in EMOTION_PROMPTS:
            # 添加情绪描述
            emotion_prompt = np.random.choice(EMOTION_PROMPTS[emotion])
            text = f"{text}，{emotion_prompt}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 生成
        if self.model_type == "emotion":
            # 情绪标签
            emotion_map = {
                "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
                "surprised": 4, "fearful": 5, "disgusted": 6,
            }
            emotion_label = emotion_map.get(emotion or "neutral", 0)

            landmarks, pose = self.model(
                input_ids,
                attention_mask,
                emotion_labels=torch.tensor([emotion_label]).to(self.device),
                emotion_intensity=torch.tensor([intensity]).to(self.device),
                sequence_length=sequence_length,
            )
        else:
            landmarks, pose = self.model(
                input_ids,
                attention_mask,
                sequence_length=sequence_length,
            )

        # 应用温度
        if temperature != 1.0:
            landmarks = landmarks / temperature
            pose = pose / temperature

        return {
            "landmarks": landmarks.squeeze(0).cpu().numpy(),
            "pose": pose.squeeze(0).cpu().numpy(),
            "text": text,
        }

    def generate_sequence(
        self,
        text_prompts: list,
        fps: int = 25,
        smooth: bool = True,
    ) -> dict:
        """
        生成关键点序列

        Args:
            text_prompts: 文本提示列表 (每个对应一帧或一段)
            fps: 帧率
            smooth: 是否平滑

        Returns:
            完整序列的关键点
        """
        all_landmarks = []
        all_poses = []

        for prompt in text_prompts:
            result = self.generate(prompt, sequence_length=1)
            all_landmarks.append(result["landmarks"])
            all_poses.append(result["pose"])

        # 合并
        landmarks = np.stack(all_landmarks, axis=0)  # (T, 21, 3)
        pose = np.stack(all_poses, axis=0)  # (T, 6)

        # 平滑
        if smooth and len(landmarks) > 1:
            window_size = 5
            for i in range(len(landmarks)):
                start = max(0, i - window_size // 2)
                end = min(len(landmarks), i + window_size // 2 + 1)
                landmarks[i] = landmarks[start:end].mean(axis=0)
                pose[i] = pose[start:end].mean(axis=0)

        return {
            "landmarks": landmarks,
            "pose": pose,
            "fps": fps,
        }


class Text2FaceWithPersonaLive:
    """
    文本到视频，集成 PersonaLive
    """

    def __init__(
        self,
        checkpoint_path: str,
        personalive_wrapper_path: str = None,
        config_path: str = None,
        device: str = "cuda",
    ):
        """
        初始化

        Args:
            checkpoint_path: Text2Face 模型路径
            personalive_wrapper_path: PersonaLive wrapper 路径
            config_path: 配置文件路径
            device: 运行设备
        """
        self.device = device

        # Text2Face 模型
        self.text2face = Text2FaceInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )

        # PersonaLive (可选)
        self.personalive = None
        if personalive_wrapper_path:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from src.wrapper import PersonaLive

                self.personalive = PersonaLive()
                print("PersonaLive wrapper loaded")
            except Exception as e:
                print(f"Warning: Could not load PersonaLive: {e}")

    def generate_video(
        self,
        text: str,
        reference_image: str,
        emotion: str = None,
        intensity: float = 0.7,
        duration: float = 3.0,
        fps: int = 25,
        output_path: str = None,
    ) -> str:
        """
        生成视频

        Args:
            text: 文本描述
            reference_image: 参考图像路径
            emotion: 情绪标签
            intensity: 情绪强度
            duration: 视频时长 (秒)
            fps: 帧率
            output_path: 输出视频路径

        Returns:
            输出视频路径
        """
        if self.personalive is None:
            raise RuntimeError("PersonaLive not available")

        # 生成关键点序列
        num_frames = int(duration * fps)

        # 为每一帧生成提示
        prompts = [text] * num_frames

        keypoints = self.text2face.generate_sequence(
            text_prompts=prompts,
            fps=fps,
            smooth=True,
        )

        # 转换为 PersonaLive 格式
        pose_sequence = []
        for i in range(num_frames):
            pose_dict = {
                "pitch": float(keypoints["pose"][i, 0]),
                "yaw": float(keypoints["pose"][i, 1]),
                "roll": float(keypoints["pose"][i, 2]),
                "t": keypoints["pose"][i, 3:6].tolist(),
                "scale": float(keypoints["pose"][i, 5]),
                "kp": torch.from_numpy(keypoints["landmarks"][i]).float(),
            }
            pose_sequence.append(pose_dict)

        # 使用 PersonaLive 生成视频
        result = self.personalive.generate_with_pose_sequence(
            reference_image=reference_image,
            pose_sequence=pose_sequence,
        )

        if output_path:
            # 保存视频
            import imageio
            imageio.mimsave(output_path, result, fps=fps)
            print(f"Video saved to {output_path}")

        return result

    def generate_stream(
        self,
        text_generator,
        reference_image: str,
        fps: int = 25,
    ):
        """
        流式生成视频

        Args:
            text_generator: 文本生成器 (yield 文本)
            reference_image: 参考图像路径
            fps: 帧率
        """
        if self.personalive is None:
            raise RuntimeError("PersonaLive not available")

        # 融合参考图像
        self.personalive.fuse_reference(reference_image)

        # 流式生成
        for text in text_generator:
            # 生成关键点
            keypoints = self.text2face.generate(text, sequence_length=1)

            # 转换格式
            pose_dict = {
                "pitch": float(keypoints["pose"][0, 0]),
                "yaw": float(keypoints["pose"][0, 1]),
                "roll": float(keypoints["pose"][0, 2]),
                "t": keypoints["pose"][0, 3:6].tolist(),
                "scale": float(keypoints["pose"][0, 5]),
                "kp": torch.from_numpy(keypoints["landmarks"][0]).float(),
            }

            # 生成帧
            frame = self.personalive.generate_with_pose(pose_dict)
            yield frame


def main():
    parser = argparse.ArgumentParser(description="Text2Face Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    parser.add_argument("--config", type=str,
                        help="Path to config file")

    parser.add_argument("--text", type=str, required=True,
                        help="Input text")

    parser.add_argument("--emotion", type=str,
                        choices=["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"],
                        help="Emotion label")

    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Emotion intensity [0, 1]")

    parser.add_argument("--duration", type=float, default=3.0,
                        help="Video duration in seconds")

    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second")

    parser.add_argument("--reference", type=str,
                        help="Reference image path (for PersonaLive)")

    parser.add_argument("--output", type=str,
                        help="Output video path")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    # 创建推理器
    inference = Text2FaceWithPersonaLive(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    if args.reference:
        # 生成视频
        output = inference.generate_video(
            text=args.text,
            reference_image=args.reference,
            emotion=args.emotion,
            intensity=args.intensity,
            duration=args.duration,
            fps=args.fps,
            output_path=args.output,
        )
        print(f"Generated {len(output)} frames")
    else:
        # 只生成关键点
        result = inference.text2face.generate(
            text=args.text,
            emotion=args.emotion,
            intensity=args.intensity,
            sequence_length=int(args.duration * args.fps),
        )

        print(f"Generated keypoints:")
        print(f"  Landmarks shape: {result['landmarks'].shape}")
        print(f"  Pose shape: {result['pose'].shape}")

        # 保存
        if args.output:
            np.savez(
                args.output,
                landmarks=result["landmarks"],
                pose=result["pose"],
            )
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
