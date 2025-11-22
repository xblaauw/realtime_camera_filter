import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class SobelEdgeFilter:
    """Sobel edge detection filter using PyTorch."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Define Sobel kernels for X and Y gradients
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)

        # Reshape for conv2d: [out_channels, in_channels, height, width]
        # We'll apply to each color channel separately
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(device)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Sobel edge detection to a frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Edge-detected frame as numpy array (H, W, C) in BGR format
        """
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

        # Convert back to numpy array [H, W, C] in BGR format
        edge_frame = edge_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        edge_frame = (edge_frame * 255).astype(np.uint8)
        edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_RGB2BGR)

        return edge_frame


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


class VideoPipeline:
    """Video processing pipeline with filters."""

    def __init__(self, source: VideoSource, filter_obj: SobelEdgeFilter):
        self.source = source
        self.filter = filter_obj
        self.props = source.get_properties()

    def process_to_file(self, output_path: Path, show_preview: bool = False):
        """
        Process video and save to file.

        Args:
            output_path: Path to save output video
            show_preview: Whether to show real-time preview window
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.props["fps"],
            (self.props["width"], self.props["height"])
        )

        frame_count = 0
        total_frames = self.props.get("frame_count", "unknown")

        print(f"Processing video...")
        print(f"Resolution: {self.props['width']}x{self.props['height']}")
        print(f"FPS: {self.props['fps']}")
        print(f"Total frames: {total_frames}")
        print(f"Output: {output_path}")

        try:
            while True:
                ret, frame = self.source.read()
                if not ret:
                    break

                # Apply filter
                processed_frame = self.filter.apply(frame)

                # Write to output
                out.write(processed_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames", end="\r")

                # Show preview if requested
                if show_preview:
                    cv2.imshow('Processed Frame', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break

            print(f"\nProcessed {frame_count} frames total")
            print(f"Saved to: {output_path}")

        finally:
            out.release()
            self.source.release()
            if show_preview:
                cv2.destroyAllWindows()

    def process_realtime(self, window_name: str = "Real-time Edge Detection"):
        """
        Process video in real-time with preview window.
        Press 'q' to quit.

        Args:
            window_name: Name of the preview window
        """
        print(f"Starting real-time processing...")
        print(f"Resolution: {self.props['width']}x{self.props['height']}")
        print(f"Press 'q' to quit")

        try:
            while True:
                ret, frame = self.source.read()
                if not ret:
                    break

                # Apply filter
                processed_frame = self.filter.apply(frame)

                # Display
                cv2.imshow(window_name, processed_frame)

                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user")
                    break

        finally:
            self.source.release()
            cv2.destroyAllWindows()
