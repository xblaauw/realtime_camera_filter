# Real-time Camera Filter

A modular video processing pipeline with multiple real-time filters for video files and webcam streams. Features Sobel edge detection, cartoon effects, thermal vision, and pixelation - all with GPU acceleration support.

![Example output with blur=3, threshold=0.0](examples/screenshot_from_feed.png)
*Real-time Sobel edge detection with `--filter sobel --blur 3 --threshold 0.0`*

## Features

- **Four built-in filters**: Sobel edge detection, cartoon/comic effect, thermal vision, and pixelation
- **Real-time webcam processing** with live preview
- **Video file processing** with audio preservation
- **GPU acceleration** via PyTorch (CUDA support)
- **Adjustable parameters** for fine-tuning each filter
- **Modular architecture** - easy to extend with custom filters
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
python webcam.py [camera_id] --filter FILTER [filter-options]
```

**Filter Options:**

**Sobel Edge Detection** (default):
```bash
python webcam.py --filter sobel
python webcam.py --filter sobel --threshold 0.03 --blur 7
```
- `--threshold` (0.0-1.0): Lower = more edges (default: 0.03)
- `--blur` (odd int): Higher = smoother edges (default: 7)

**Cartoon/Comic Effect**:
```bash
python webcam.py --filter cartoon
python webcam.py --filter cartoon --color-levels 6 --edge-threshold 0.15
```
- `--edge-threshold` (0.0-1.0): Lower = more edges (default: 0.2)
- `--color-levels` (4-16): Lower = more cartoony (default: 8)
- `--blur-d` (5-15): Higher = smoother colors (default: 9)

**Thermal Vision**:
```bash
python webcam.py --filter thermal
python webcam.py --filter thermal --colormap hot --contrast 2.0
```
- `--colormap`: jet, hot, inferno, cool, bone, viridis (default: jet)
- `--brightness` (0.5-2.0): Higher = brighter (default: 1.0)
- `--contrast` (0.5-3.0): Higher = more contrast (default: 1.5)

**Pixelation**:
```bash
python webcam.py --filter pixel
python webcam.py --filter pixel --pixel-size 32 --smooth
```
- `--pixel-size` (4-64): Higher = more pixelated (default: 16)
- `--smooth`: Smooth pixel edges instead of hard edges

**Controls:**
- Press `q` to quit the preview window

### Video File Processing

Process video files while preserving audio:

```bash
python process_input.py INPUT_VIDEO [OUTPUT_VIDEO] --filter FILTER [filter-options]
```

**Examples:**
```bash
# Sobel edge detection
python process_input.py input/video.mp4 --filter sobel --threshold 0.05

# Cartoon effect
python process_input.py input/video.mp4 --filter cartoon --color-levels 6

# Thermal vision with hot colormap
python process_input.py input/video.mp4 --filter thermal --colormap hot

# Pixelation with smooth edges
python process_input.py input/video.mp4 --filter pixel --pixel-size 24 --smooth

# Show preview while processing
python process_input.py input/video.mp4 --filter cartoon --preview

# Process without preserving audio
python process_input.py input/video.mp4 --filter sobel --no-audio
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
├── pipeline.py           # Core pipeline and filter classes
├── webcam.py             # Webcam processing script
├── process_input.py      # Video file processing script
├── main.py               # Legacy main script (can be used as reference)
├── input/                # Place input videos here
├── output/               # Processed videos saved here
├── examples/             # Example screenshots
└── README.md             # This file
```

## How It Works

### Architecture

The modular design separates concerns:

- **Filter Classes**: Each filter implements an `apply(frame) -> frame` method
  - `SobelEdgeFilter`: PyTorch-based edge detection
  - `CartoonFilter`: Bilateral filter + edge detection + color quantization
  - `ThermalVisionFilter`: Grayscale conversion + color mapping
  - `PixelationFilter`: Downscale + upscale with interpolation

- **Video Sources**: Abstract interface for frame capture
  - `VideoFileSource`: Reads MP4/video files
  - `WebcamSource`: Captures from webcam

- **VideoPipeline**: Orchestrates frame processing
  - `process_to_file()`: For video files (preserves audio via ffmpeg)
  - `process_realtime()`: For live webcam streams

### Adding Custom Filters

Create your own filter by implementing the `apply()` method:

```python
class MyCustomFilter:
    def __init__(self, param1=default1, param2=default2):
        self.param1 = param1
        self.param2 = param2

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # Your processing logic here
        # frame is (H, W, C) BGR format
        processed = your_processing(frame)
        return processed

# Use it
from pipeline import WebcamSource, VideoPipeline

filter_obj = MyCustomFilter(param1=value1)
source = WebcamSource(0)
pipeline = VideoPipeline(source, filter_obj)
pipeline.process_realtime()
```

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
Works seamlessly with virtual camera apps like DroidCam:
```bash
python webcam.py 0 --filter thermal --colormap inferno
```

### Batch Processing
Process multiple videos with the same filter:
```bash
for video in input/*.mp4; do
    python process_input.py "$video" --filter cartoon --color-levels 6
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

Contributions welcome! Feel free to:
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
