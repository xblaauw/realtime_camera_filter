"""Sobel edge detection filter."""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .base import BaseFilter


class SobelEdgeFilter(BaseFilter):
    """Sobel edge detection filter using PyTorch."""

    name = "sobel"
    description = "PyTorch-based Sobel edge detection"

    def __init__(self, threshold: float = 0.03, blur_kernel_size: int = 7, device: str = None):
        """
        Initialize Sobel edge filter.

        Args:
            threshold: Minimum edge strength (0.0-1.0). Lower = more edges, Higher = fewer edges
            blur_kernel_size: Size of Gaussian blur kernel (odd number, 0 to disable)
            device: Device to run on (cuda/cpu). Auto-detect if None.
        """
        super().__init__()
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Define Sobel kernels for X and Y gradients
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)

        # Reshape for conv2d: [out_channels, in_channels, height, width]
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(self.device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(self.device)

    @classmethod
    def get_parameters(cls):
        return {
            'threshold': {
                'type': float,
                'default': 0.03,
                'help': 'Edge detection threshold (0.0-1.0). Lower = more edges',
                'range': (0.0, 1.0)
            },
            'blur': {
                'type': int,
                'default': 7,
                'help': 'Gaussian blur kernel size (odd number). Higher = smoother',
                'range': (1, 21)
            }
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection to a frame."""
        # Apply Gaussian blur if enabled
        if self.blur_kernel_size > 0:
            frame = cv2.GaussianBlur(frame, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # Convert BGR to RGB and normalize to [0, 1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0

        # Rearrange to [C, H, W] and add batch dimension [1, C, H, W]
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Process each channel separately
        edges = []
        for c in range(3):
            channel = frame_tensor[:, c:c+1, :, :]

            # Apply Sobel operators
            grad_x = F.conv2d(channel, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)

            # Compute gradient magnitude
            edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            edges.append(edge)

        # Combine channels
        edge_tensor = torch.cat(edges, dim=1)

        # Normalize to [0, 1]
        edge_tensor = edge_tensor / edge_tensor.max() if edge_tensor.max() > 0 else edge_tensor

        # Apply threshold to filter out weak edges
        if self.threshold > 0:
            edge_tensor = torch.where(edge_tensor > self.threshold, edge_tensor, torch.zeros_like(edge_tensor))

        # Convert back to numpy array [H, W, C] in BGR format
        edge_frame = edge_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        edge_frame = (edge_frame * 255).astype(np.uint8)
        edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_RGB2BGR)

        return edge_frame
