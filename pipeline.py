import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import shutil


class SobelEdgeFilter:
    """Sobel edge detection filter using PyTorch."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 threshold: float = 0.1, blur_kernel_size: int = 3):
        """
        Initialize Sobel edge filter.

        Args:
            device: Device to run on (cuda/cpu)
            threshold: Minimum edge strength (0.0-1.0). Lower = more edges, Higher = fewer edges
            blur_kernel_size: Size of Gaussian blur kernel (odd number, 0 to disable)
        """
        self.device = device
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size

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
        # Apply Gaussian blur if enabled (reduces noise and makes edges smoother)
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


class CartoonFilter:
    """Cartoon/comic book filter using bilateral filtering and edge detection."""

    def __init__(self, edge_threshold: float = 0.2, color_levels: int = 8, blur_d: int = 9):
        """
        Initialize cartoon filter.

        Args:
            edge_threshold: Edge detection sensitivity (0.0-1.0). Lower = more edges
            color_levels: Number of color levels for quantization (4-16). Lower = more cartoony
            blur_d: Bilateral filter diameter (odd number, 5-15). Higher = smoother colors
        """
        self.edge_threshold = edge_threshold
        self.color_levels = color_levels
        self.blur_d = blur_d

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply cartoon filter to a frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Cartoonified frame as numpy array (H, W, C) in BGR format
        """
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
        # Reduce to fewer colors for cartoon effect
        color = color // (256 // self.color_levels) * (256 // self.color_levels)

        # Step 4: Combine edges with colored image
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_colored)

        return cartoon


class ThermalVisionFilter:
    """Thermal/night vision filter with color mapping."""

    def __init__(self, colormap: int = cv2.COLORMAP_JET, brightness: float = 1.0, contrast: float = 1.5):
        """
        Initialize thermal vision filter.

        Args:
            colormap: OpenCV colormap (COLORMAP_JET, COLORMAP_HOT, COLORMAP_INFERNO, etc.)
            brightness: Brightness adjustment (0.5-2.0). Higher = brighter
            contrast: Contrast adjustment (0.5-3.0). Higher = more contrast
        """
        self.colormap = colormap
        self.brightness = brightness
        self.contrast = contrast

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply thermal vision filter to a frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Thermal-style frame as numpy array (H, W, C) in BGR format
        """
        # Convert to grayscale to simulate thermal intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply brightness and contrast adjustments
        gray = cv2.convertScaleAbs(gray, alpha=self.contrast, beta=(self.brightness - 1.0) * 50)

        # Apply colormap for thermal effect
        thermal = cv2.applyColorMap(gray, self.colormap)

        return thermal


class PixelationFilter:
    """Pixelation/retro filter."""

    def __init__(self, pixel_size: int = 16, smooth: bool = False):
        """
        Initialize pixelation filter.

        Args:
            pixel_size: Size of each pixel block (4-64). Higher = more pixelated
            smooth: Whether to smooth the pixelated output (creates softer blocks)
        """
        self.pixel_size = pixel_size
        self.smooth = smooth

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply pixelation filter to a frame.

        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format

        Returns:
            Pixelated frame as numpy array (H, W, C) in BGR format
        """
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

    def __init__(self, source: VideoSource, filter_obj: SobelEdgeFilter, source_path: Optional[Path] = None):
        self.source = source
        self.filter = filter_obj
        self.props = source.get_properties()
        self.source_path = source_path

    def process_to_file(self, output_path: Path, show_preview: bool = False, preserve_audio: bool = True):
        """
        Process video and save to file.

        Args:
            output_path: Path to save output video
            show_preview: Whether to show real-time preview window
            preserve_audio: Whether to preserve audio from source video (if available)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file for video without audio
        temp_video = None
        if preserve_audio and self.source_path:
            temp_video = output_path.parent / f"temp_{output_path.name}"
            video_output = temp_video
        else:
            video_output = output_path

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(video_output),
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

        finally:
            out.release()
            self.source.release()
            if show_preview:
                cv2.destroyAllWindows()

        # Merge with audio if requested and source has audio
        if preserve_audio and self.source_path and temp_video:
            try:
                print("Merging audio from source video...")
                # Use ffmpeg directly to copy audio stream from source
                result = subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(temp_video),
                    '-i', str(self.source_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    str(output_path)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    # Remove temporary file
                    temp_video.unlink()
                    print(f"Saved with audio to: {output_path}")
                else:
                    # Check if source has no audio stream
                    if "does not contain any stream" in result.stderr or "Stream map" in result.stderr:
                        shutil.move(str(temp_video), str(output_path))
                        print(f"No audio in source. Saved to: {output_path}")
                    else:
                        raise Exception(f"ffmpeg failed: {result.stderr}")

            except Exception as e:
                print(f"Warning: Could not merge audio ({e})")
                # Fallback: just rename temp file
                if temp_video.exists():
                    shutil.move(str(temp_video), str(output_path))
                print(f"Saved without audio to: {output_path}")
        else:
            print(f"Saved to: {output_path}")

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
