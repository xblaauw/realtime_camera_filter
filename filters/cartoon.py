"""Cartoon/comic book filter."""

import cv2
import numpy as np
from .base import BaseFilter


class CartoonFilter(BaseFilter):
    """Cartoon/comic book filter using bilateral filtering and edge detection."""

    name = "cartoon"
    description = "Cartoon/comic book effect with edge detection and color quantization"

    def __init__(self, edge_threshold: float = 0.2, color_levels: int = 8, blur_d: int = 9):
        """
        Initialize cartoon filter.

        Args:
            edge_threshold: Edge detection sensitivity (0.0-1.0). Lower = more edges
            color_levels: Number of color levels for quantization (4-16). Lower = more cartoony
            blur_d: Bilateral filter diameter (odd number, 5-15). Higher = smoother colors
        """
        super().__init__()
        self.edge_threshold = edge_threshold
        self.color_levels = color_levels
        self.blur_d = blur_d

    @classmethod
    def get_parameters(cls):
        return {
            'edge-threshold': {
                'type': float,
                'default': 0.2,
                'help': 'Edge sensitivity (0.0-1.0). Lower = more edges',
                'range': (0.0, 1.0)
            },
            'color-levels': {
                'type': int,
                'default': 8,
                'help': 'Color quantization levels (4-16). Lower = more cartoony',
                'range': (4, 16)
            },
            'blur-d': {
                'type': int,
                'default': 9,
                'help': 'Bilateral filter diameter (5-15). Higher = smoother',
                'range': (5, 15)
            }
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply cartoon filter to a frame."""
        # Step 1: Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=int(self.edge_threshold * 10)
        )

        # Step 2: Bilateral filter for color smoothing while preserving edges
        color = cv2.bilateralFilter(frame, self.blur_d, sigmaColor=300, sigmaSpace=300)

        # Step 3: Color quantization
        color = color // (256 // self.color_levels) * (256 // self.color_levels)

        # Step 4: Combine edges with colored image
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_colored)

        return cartoon
