"""Base filter class for all video filters."""

import numpy as np
from typing import Dict, Any


class BaseFilter:
    """Base class for video filters."""

    name = "base"
    description = "Base filter class"

    def __init__(self, **kwargs):
        """Initialize filter with parameters."""
        pass

    @classmethod
    def get_parameters(cls) -> Dict[str, Any]:
        """
        Return filter parameters with their metadata.

        Returns:
            Dictionary mapping parameter name to metadata dict with keys:
            - type: Parameter type (float, int, str, bool)
            - default: Default value
            - help: Help text
            - choices: Optional list of valid choices
            - range: Optional tuple of (min, max) for numeric types
        """
        return {}

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply filter to a frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Filtered frame as numpy array (H, W, C) in BGR format
        """
        raise NotImplementedError("Filter must implement apply() method")
