"""Output destinations for processed video."""

import cv2
import numpy as np
import subprocess


class VirtualCameraOutput:
    """Virtual camera output using v4l2loopback and ffmpeg."""

    def __init__(self, device: str, width: int, height: int):
        self.device = device
        self.width = width
        self.height = height

        self.process = subprocess.Popen([
            'ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', '30',
            '-i', '-',
            '-pix_fmt', 'yuv420p',
            '-f', 'v4l2',
            device
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray):
        """Write BGR frame to virtual camera."""
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()

    def release(self):
        """Close the virtual camera."""
        if self.process:
            self.process.stdin.close()
            self.process.wait()


__all__ = ['VirtualCameraOutput']
