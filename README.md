# Real-time Camera Filter

A modular video processing pipeline with multiple real-time filters for video files and webcam streams. Features Sobel edge detection, cartoon effects, thermal vision, and pixelation - all with GPU acceleration support.

![Example output with blur=3, threshold=0.0](examples/screenshot_from_feed.png)
*Real-time Sobel edge detection with `--filter sobel --blur 3 --threshold 0.0`*

## Features

- **Four built-in filters**: Sobel edge detection, cartoon/comic effect, thermal vision, and pixelation
- **Real-time webcam processing** with live preview
- **Virtual camera output** for video conferencing (Google Meet, Teams, Discord, etc.)
- **Video file processing** with audio preservation
- **GPU acceleration** via PyTorch (CUDA support)
- **Adjustable parameters** for fine-tuning each filter
- **Modular architecture** with filter registry pattern
- Command-line interface for both webcam and video processing

## Installation

### Prerequisites

- Python 3.12 or higher
- ffmpeg (for audio processing)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xblaauw/realtime_camera_filter
cd realtime_camera_filter
```

2. Create get correct python version, create .venv & install dependencies using `uv`:
```bash
uv sync
```

## Usage

### Webcam Processing

Process your webcam feed in real-time with various filters:

```bash
uv run webcam.py [camera_id] --filter FILTER [filter-options]
```

**Filter Options:**

**Sobel Edge Detection** (default):
```bash
uv run webcam.py --filter sobel
uv run webcam.py --filter sobel --threshold 0.03 --blur 7
```
- `--threshold` (0.0-1.0): Lower = more edges (default: 0.03)
- `--blur` (odd int): Higher = smoother edges (default: 7)

**Cartoon/Comic Effect**:
```bash
uv run webcam.py --filter cartoon
uv run webcam.py --filter cartoon --color-levels 6 --edge-threshold 0.15
```
- `--edge-threshold` (0.0-1.0): Lower = more edges (default: 0.2)
- `--color-levels` (4-16): Lower = more cartoony (default: 8)
- `--blur-d` (5-15): Higher = smoother colors (default: 9)

**Thermal Vision**:
```bash
uv run webcam.py --filter thermal
uv run webcam.py --filter thermal --colormap hot --contrast 2.0
```
- `--colormap`: jet, hot, inferno, cool, bone, viridis (default: jet)
- `--brightness` (0.5-2.0): Higher = brighter (default: 1.0)
- `--contrast` (0.5-3.0): Higher = more contrast (default: 1.5)

**Pixelation**:
```bash
uv run webcam.py --filter pixel
uv run webcam.py --filter pixel --pixel-size 32 --smooth
```
- `--pixel-size` (4-64): Higher = more pixelated (default: 16)
- `--smooth`: Smooth pixel edges instead of hard edges

**Controls:**
- Press `q` to quit the preview window

### Virtual Camera Output

Output filtered video to a virtual camera for use in video calls (Google Meet, Teams, Discord, etc.):

**Setup (one-time):**

1. Install v4l2loopback kernel module:
```bash
sudo apt install v4l2loopback-dkms
```

2. Create virtual camera devices (2 devices: one for OBS, one for filtered output):
```bash
sudo modprobe v4l2loopback devices=2 video_nr=0,2 card_label="OBS,Filtered" exclusive_caps=1,1
```

3. Start OBS and enable "Start Virtual Camera" to activate `/dev/video0`

**Usage:**

```bash
# Read from OBS virtual camera (video0), apply filter, output to video2
uv run webcam.py 0 --filter thermal --virtual-cam /dev/video2

# Use Sobel edge detection
uv run webcam.py 0 --filter sobel --blur 5 --threshold 0.05 --virtual-cam /dev/video2

# Use cartoon filter
uv run webcam.py 0 --filter cartoon --color-levels 6 --virtual-cam /dev/video2
```

Then select "Filtered" camera in your video conferencing app.

**Persistence:**

To make v4l2loopback settings persist across reboots:
```bash
echo "options v4l2loopback devices=2 video_nr=0,2 card_label=\"OBS,Filtered\" exclusive_caps=1,1" | sudo tee /etc/modprobe.d/v4l2loopback.conf
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
```

**Troubleshooting:**
- If `/dev/video2` doesn't exist, reload the module: `sudo rmmod v4l2loopback && sudo modprobe v4l2loopback devices=2 video_nr=0,2 card_label="OBS,Filtered" exclusive_caps=1,1`
- If module is busy: Close all apps using the virtual cameras (Firefox, OBS, etc.), then reload
- OBS must be restarted after reloading v4l2loopback module

### Video File Processing

Process video files while preserving audio:

```bash
uv run process_input.py INPUT_VIDEO [OUTPUT_VIDEO] --filter FILTER [filter-options]
```

**Examples:**
```bash
# Sobel edge detection
uv run process_input.py input/video.mp4 --filter sobel --threshold 0.05

# Cartoon effect
uv run process_input.py input/video.mp4 --filter cartoon --color-levels 6

# Thermal vision with hot colormap
uv run process_input.py input/video.mp4 --filter thermal --colormap hot

# Pixelation with smooth edges
uv run process_input.py input/video.mp4 --filter pixel --pixel-size 24 --smooth

# Show preview while processing
uv run process_input.py input/video.mp4 --filter cartoon --preview

# Process without preserving audio
uv run process_input.py input/video.mp4 --filter sobel --no-audio
```

**General Parameters:**
- `INPUT_VIDEO`: Path to input video file (required)
- `OUTPUT_VIDEO`: Path to output video file (optional, defaults to `output/<filter>_<input_name>`)
- `--preview`: Show real-time preview during processing
- `--no-audio`: Don't preserve audio from source video

## Available Filters

### 1. Sobel Edge Detection
PyTorch-based edge detection using Sobel operators with Gaussian blur preprocessing.

**Use cases:** Line art, artistic effects, feature extraction

### 2. Cartoon Filter
Combines bilateral filtering, edge detection, and color quantization for a comic book appearance.

**Use cases:** Stylized content, artistic videos, comic-style streams

### 3. Thermal Vision
Converts to grayscale and applies color mapping to simulate thermal/night vision cameras.

**Use cases:** Atmospheric effects, surveillance aesthetic, gaming overlays

### 4. Pixelation
Retro-style pixelation with optional smoothing.

**Use cases:** Retro gaming aesthetic, privacy blurring, artistic effects

## Project Structure

```
realtime_camera_filter/
├── filters/              # Filter modules (auto-discovered)
│   ├── __init__.py       # Filter registry
│   ├── base.py           # BaseFilter class
│   ├── sobel.py          # SobelEdgeFilter
│   ├── cartoon.py        # CartoonFilter
│   ├── thermal.py        # ThermalVisionFilter
│   └── pixelation.py     # PixelationFilter
├── sources/              # Video source modules
│   └── __init__.py       # VideoSource, VideoFileSource, WebcamSource
├── outputs/              # Output destination modules
│   └── __init__.py       # VirtualCameraOutput (v4l2loopback via ffmpeg)
├── pipeline.py           # VideoPipeline orchestration
├── webcam.py             # Webcam processing CLI (auto-discovers filters)
├── process_input.py      # Video file processing CLI (auto-discovers filters)
├── main.py               # Legacy main script (can be used as reference)
├── input/                # Place input videos here
├── output/               # Processed videos saved here
├── examples/             # Example screenshots
└── README.md             # This file
```

## How It Works

### Architecture

The modular design uses a **filter registry pattern**:

- **Filter Registry** (`filters/__init__.py`): Auto-discovers and manages all filters
  - `list_filters()`: Get available filter names
  - `get_filter_info()`: Get filter metadata and parameters
  - `create_filter()`: Instantiate a filter with parameters

- **Filter Classes** (`filters/`): Each filter inherits from `BaseFilter`
  - `SobelEdgeFilter`: PyTorch-based edge detection
  - `CartoonFilter`: Bilateral filter + edge detection + color quantization
  - `ThermalVisionFilter`: Grayscale conversion + color mapping
  - `PixelationFilter`: Downscale + upscale with interpolation
  - Filters define their own parameters via `get_parameters()`

- **Video Sources** (`sources/`): Abstract interface for frame capture
  - `VideoFileSource`: Reads MP4/video files
  - `WebcamSource`: Captures from webcam

- **VideoPipeline** (`pipeline.py`): Orchestrates frame processing
  - `process_to_file()`: For video files (preserves audio via ffmpeg)
  - `process_realtime()`: For live webcam streams

### Design Notes

- CLI arguments are generated from filter metadata
- Filters are discovered via the registry
- Adding filters doesn't require changes to `webcam.py` or `process_input.py`

### Adding Custom Filters

Adding a new filter:

**Step 1:** Create your filter file in `filters/`:

```python
# filters/myfilter.py
import cv2
import numpy as np
from .base import BaseFilter

class MyCustomFilter(BaseFilter):
    name = "myfilter"
    description = "My amazing custom filter"

    def __init__(self, param1: int = 10, param2: float = 0.5):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    @classmethod
    def get_parameters(cls):
        return {
            'param1': {
                'type': int,
                'default': 10,
                'help': 'First parameter (1-100)',
                'range': (1, 100)
            },
            'param2': {
                'type': float,
                'default': 0.5,
                'help': 'Second parameter (0.0-1.0)',
                'range': (0.0, 1.0)
            }
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # Your filter logic here
        # frame is (H, W, C) in BGR format
        processed = your_processing(frame, self.param1, self.param2)
        return processed
```

**Step 2:** Register it in `filters/__init__.py`:

```python
from .myfilter import MyCustomFilter

FILTERS = {
    'sobel': SobelEdgeFilter,
    'cartoon': CartoonFilter,
    'thermal': ThermalVisionFilter,
    'pixel': PixelationFilter,
    'myfilter': MyCustomFilter,  # Add this line
}
```

**Step 3:** Use the filter:

```bash
uv run webcam.py --filter myfilter --param1 20 --param2 0.8
```

The CLI scripts generate arguments from the filter's `get_parameters()` method.

## Performance

- **GPU Acceleration**: Sobel filter automatically uses CUDA if available
- **Real-time Processing**: Optimized for 30+ FPS on modern hardware
- **Memory Efficient**: Processes frames individually, no full video loading
- **Filter Speed**: Thermal and Pixelation are fastest; Cartoon is slowest (bilateral filter)

## Requirements

- `torch >= 2.9.1`
- `torchvision >= 0.24.1`
- `opencv-python >= 4.12.0`
- `numpy >= 2.2.6`
- `ffmpeg` (system package for audio handling)

## Examples

### Webcam with DroidCam
Using virtual camera apps like DroidCam:
```bash
uv run webcam.py 0 --filter thermal --colormap inferno
```

### Batch Processing
Process multiple videos with the same filter:
```bash
for video in input/*.mp4; do
    uv run process_input.py "$video" --filter cartoon --color-levels 6
done
```

### Chain Filters (Advanced)
Create a custom filter that combines multiple effects:
```python
from pipeline import SobelEdgeFilter, PixelationFilter

class ChainedFilter:
    def __init__(self):
        self.pixelate = PixelationFilter(pixel_size=8)
        self.edges = SobelEdgeFilter(threshold=0.1)

    def apply(self, frame):
        frame = self.pixelate.apply(frame)
        frame = self.edges.apply(frame)
        return frame
```

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions accepted:
- Add new filters (style transfer, depth estimation, etc.)
- Improve performance
- Add features (output to virtual camera, filter combinations)
- Fix bugs

## Troubleshooting

**Q: Webcam not working?**
- Check camera ID with `ls /dev/video*` (Linux) or try different IDs
- Ensure no other application is using the webcam

**Q: Audio not preserved?**
- Ensure ffmpeg is installed: `ffmpeg -version`
- Check source video has audio: `ffprobe input.mp4`

**Q: Slow performance?**
- Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Try faster filters (thermal, pixel) instead of cartoon
- Reduce resolution or blur kernel sizes

**Q: Cartoon filter too slow?**
- Reduce `--blur-d` parameter
- Reduce `--color-levels` for faster quantization
- The bilateral filter is computationally expensive

**Q: Want different thermal colors?**
- Try different colormaps: `--colormap hot`, `--colormap inferno`, `--colormap cool`
- Adjust contrast and brightness for better visibility

## Acknowledgments

Built with PyTorch, OpenCV, and ffmpeg.
