"""Video source management."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoSource:
    """Abstract base for video sources."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame. Returns (success, frame)."""
        raise NotImplementedError

    def release(self):
        """Release resources."""
        raise NotImplementedError

    def get_properties(self) -> dict:
        """Get video properties (fps, width, height)."""
        raise NotImplementedError


class VideoFileSource(VideoSource):
    """Video file source."""

    def __init__(self, video_path: Path):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self):
        self.cap.release()

    def get_properties(self) -> dict:
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }


class WebcamSource(VideoSource):
    """Webcam source."""

    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open webcam: {camera_id}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self):
        self.cap.release()

    def get_properties(self) -> dict:
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }


__all__ = ['VideoSource', 'VideoFileSource', 'WebcamSource']
