"""
关键点提取工具

从视频/图像中提取 LivePortrait 格式的面部关键点:
- 21个3D关键点
- 6D头部姿态 (pitch, yaw, roll, tx, ty, scale)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from tqdm import tqdm
import sys

# 添加 PersonaLive 路径
PERSONALIVE_PATH = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(PERSONALIVE_PATH))

try:
    from src.liveportrait.motion_extractor import LivePortraitMotionExtractor
except ImportError:
    print("Warning: Cannot import LivePortraitMotionExtractor")
    print("Please ensure PersonaLive is properly installed")


class KeypointExtractor:
    """
    面部关键点提取器

    使用 LivePortrait 的运动提取器获取:
        - 21个3D面部关键点
        - 头部姿态参数
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        初始化提取器

        Args:
            model_path: LivePortrait 模型路径
            device: 运行设备
        """
        self.device = device

        try:
            self.extractor = LivePortraitMotionExtractor(
                model_path=model_path,
                device=device,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize LivePortraitMotionExtractor: {e}")
            self.extractor = None

    def extract_from_image(
        self,
        image: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        从单张图像提取关键点

        Args:
            image: (H, W, 3) BGR 图像

        Returns:
            dict containing:
                - landmarks: (21, 3) 3D 关键点
                - pitch: 俯仰角
                - yaw: 偏航角
                - roll: 翻滚角
                - t: (3,) 平移向量
                - scale: 缩放因子
        """
        if self.extractor is None:
            # 返回默认值用于测试
            return {
                "landmarks": np.zeros((21, 3), dtype=np.float32),
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "t": np.zeros(3, dtype=np.float32),
                "scale": 1.0,
            }

        try:
            # 提取关键点
            result = self.extractor.extract(image)

            return {
                "landmarks": result["kp"].cpu().numpy(),  # (21, 3)
                "pitch": float(result["pitch"]),
                "yaw": float(result["yaw"]),
                "roll": float(result["roll"]),
                "t": result["t"].cpu().numpy(),  # (3,)
                "scale": float(result["scale"]),
            }
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return {
                "landmarks": np.zeros((21, 3), dtype=np.float32),
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "t": np.zeros(3, dtype=np.float32),
                "scale": 1.0,
            }

    def extract_from_video(
        self,
        video_path: str,
        fps: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        从视频提取关键点序列

        Args:
            video_path: 视频文件路径
            fps: 提取帧率 (None 表示原始帧率)
            max_frames: 最大帧数

        Returns:
            List of keypoint dicts
        """
        cap = cv2.VideoCapture(video_path)
        keypoints_list = []

        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算采样间隔
        if fps is not None:
            frame_interval = int(original_fps / fps)
        else:
            frame_interval = 1

        frame_count = 0
        extracted_count = 0

        with tqdm(total=min(total_frames, max_frames or total_frames), desc=f"Extracting {Path(video_path).name}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # 采样
                if frame_count % frame_interval == 0:
                    kp = self.extract_from_image(frame)
                    keypoints_list.append(kp)
                    extracted_count += 1
                    pbar.update(1)

                    if max_frames is not None and extracted_count >= max_frames:
                        break

                frame_count += 1

        cap.release()
        return keypoints_list

    def smooth_keypoints(
        self,
        keypoints_list: List[Dict],
        window_size: int = 5,
    ) -> List[Dict]:
        """
        时序平滑关键点

        Args:
            keypoints_list: 关键点列表
            window_size: 滑动窗口大小

        Returns:
            平滑后的关键点列表
        """
        n = len(keypoints_list)
        smoothed = []

        for i in range(n):
            # 计算窗口范围
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)

            # 平均关键点
            window = keypoints_list[start:end]

            smoothed_kp = {
                "landmarks": np.mean([kp["landmarks"] for kp in window], axis=0),
                "pitch": np.mean([kp["pitch"] for kp in window]),
                "yaw": np.mean([kp["yaw"] for kp in window]),
                "roll": np.mean([kp["roll"] for kp in window]),
                "t": np.mean([kp["t"] for kp in window], axis=0),
                "scale": np.mean([kp["scale"] for kp in window]),
            }

            smoothed.append(smoothed_kp)

        return smoothed


def save_keypoints(
    keypoints: Dict or List[Dict],
    output_path: str,
):
    """
    保存关键点到文件

    Args:
        keypoints: 单帧或序列关键点
        output_path: 输出文件路径 (.npz 或 .json)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".npz":
        # NumPy 格式
        if isinstance(keypoints, list):
            # 序列
            data = {
                "landmarks": np.stack([kp["landmarks"] for kp in keypoints]),
                "pitch": np.array([kp["pitch"] for kp in keypoints]),
                "yaw": np.array([kp["yaw"] for kp in keypoints]),
                "roll": np.array([kp["roll"] for kp in keypoints]),
                "t": np.stack([kp["t"] for kp in keypoints]),
                "scale": np.array([kp["scale"] for kp in keypoints]),
            }
        else:
            # 单帧
            data = keypoints

        np.savez_compressed(output_path, **data)

    elif output_path.suffix == ".json":
        # JSON 格式 (需要转换)
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        with open(output_path, "w") as f:
            json.dump(keypoints, f, default=to_serializable, indent=2)

    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}")


def load_keypoints(
    input_path: str,
) -> Dict or List[Dict]:
    """
    从文件加载关键点

    Args:
        input_path: 输入文件路径

    Returns:
        关键点数据
    """
    input_path = Path(input_path)

    if input_path.suffix == ".npz":
        data = np.load(input_path)

        # 判断是单帧还是序列
        if data["landmarks"].ndim == 2:
            # 单帧: (21, 3)
            return {
                "landmarks": data["landmarks"],
                "pitch": float(data["pitch"]),
                "yaw": float(data["yaw"]),
                "roll": float(data["roll"]),
                "t": data["t"],
                "scale": float(data["scale"]),
            }
        else:
            # 序列: (T, 21, 3)
            n = data["landmarks"].shape[0]
            return [
                {
                    "landmarks": data["landmarks"][i],
                    "pitch": float(data["pitch"][i]),
                    "yaw": float(data["yaw"][i]),
                    "roll": float(data["roll"][i]),
                    "t": data["t"][i],
                    "scale": float(data["scale"][i]),
                }
                for i in range(n)
            ]

    elif input_path.suffix == ".json":
        with open(input_path, "r") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


if __name__ == "__main__":
    # 测试代码
    extractor = KeypointExtractor(device="cpu")

    # 测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    kp = extractor.extract_from_image(test_image)
    print("Single frame keypoints:")
    print(f"  Landmarks shape: {kp['landmarks'].shape}")
    print(f"  Pose: pitch={kp['pitch']:.2f}, yaw={kp['yaw']:.2f}, roll={kp['roll']:.2f}")
