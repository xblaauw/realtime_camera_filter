"""Thermal/night vision filter."""

import cv2
import numpy as np
from .base import BaseFilter


# Colormap name to cv2 constant mapping
COLORMAP_OPTIONS = {
    'jet': cv2.COLORMAP_JET,
    'hot': cv2.COLORMAP_HOT,
    'inferno': cv2.COLORMAP_INFERNO,
    'cool': cv2.COLORMAP_COOL,
    'bone': cv2.COLORMAP_BONE,
    'viridis': cv2.COLORMAP_VIRIDIS
}


class ThermalVisionFilter(BaseFilter):
    """Thermal/night vision filter with color mapping."""

    name = "thermal"
    description = "Thermal/night vision effect with color mapping"

    def __init__(self, colormap: str = 'jet', brightness: float = 1.0, contrast: float = 1.5):
        """
        Initialize thermal vision filter.

        Args:
            colormap: Color scheme name (jet, hot, inferno, cool, bone, viridis)
            brightness: Brightness adjustment (0.5-2.0). Higher = brighter
            contrast: Contrast adjustment (0.5-3.0). Higher = more contrast
        """
        super().__init__()
        self.colormap = COLORMAP_OPTIONS.get(colormap, cv2.COLORMAP_JET)
        self.colormap_name = colormap
        self.brightness = brightness
        self.contrast = contrast

    @classmethod
    def get_parameters(cls):
        return {
            'colormap': {
                'type': str,
                'default': 'jet',
                'help': 'Color scheme',
                'choices': list(COLORMAP_OPTIONS.keys())
            },
            'brightness': {
                'type': float,
                'default': 1.0,
                'help': 'Brightness (0.5-2.0). Higher = brighter',
                'range': (0.5, 2.0)
            },
            'contrast': {
                'type': float,
                'default': 1.5,
                'help': 'Contrast (0.5-3.0). Higher = more contrast',
                'range': (0.5, 3.0)
            }
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply thermal vision filter to a frame."""
        # Convert to grayscale to simulate thermal intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply brightness and contrast adjustments
        gray = cv2.convertScaleAbs(gray, alpha=self.contrast, beta=(self.brightness - 1.0) * 50)

        # Apply colormap for thermal effect
        thermal = cv2.applyColorMap(gray, self.colormap)

        return thermal
