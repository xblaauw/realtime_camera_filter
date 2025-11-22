"""Pixelation/retro filter."""

import cv2
import numpy as np
from .base import BaseFilter


class PixelationFilter(BaseFilter):
    """Pixelation/retro filter."""

    name = "pixel"
    description = "Retro-style pixelation effect"

    def __init__(self, pixel_size: int = 16, smooth: bool = False):
        """
        Initialize pixelation filter.

        Args:
            pixel_size: Size of each pixel block (4-64). Higher = more pixelated
            smooth: Whether to smooth the pixelated output (creates softer blocks)
        """
        super().__init__()
        self.pixel_size = pixel_size
        self.smooth = smooth

    @classmethod
    def get_parameters(cls):
        return {
            'pixel-size': {
                'type': int,
                'default': 16,
                'help': 'Block size (4-64). Higher = more pixelated',
                'range': (4, 64)
            },
            'smooth': {
                'type': bool,
                'default': False,
                'help': 'Use smooth pixel edges instead of hard edges'
            }
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply pixelation filter to a frame."""
        height, width = frame.shape[:2]

        # Downscale to create pixelation
        small_width = max(1, width // self.pixel_size)
        small_height = max(1, height // self.pixel_size)

        # Resize down
        temp = cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_LINEAR)

        # Resize back up
        if self.smooth:
            # Use linear interpolation for smoother pixels
            pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            # Use nearest neighbor for hard pixel edges
            pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        return pixelated
