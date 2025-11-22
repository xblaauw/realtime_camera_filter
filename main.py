from pathlib import Path
from pipeline import SobelEdgeFilter, VideoFileSource, WebcamSource, VideoPipeline


def process_video_file(input_path: Path, output_path: Path, show_preview: bool = False):
    """Process a video file with Sobel edge detection."""
    print(f"Loading video from: {input_path}")

    # Initialize filter and source
    filter_obj = SobelEdgeFilter()
    source = VideoFileSource(input_path)

    # Create and run pipeline (pass source_path for audio preservation)
    pipeline = VideoPipeline(source, filter_obj, source_path=input_path)
    pipeline.process_to_file(output_path, show_preview=show_preview, preserve_audio=True)


def process_webcam(camera_id: int = 0, threshold: float = 0.05, blur: int = 5):
    """
    Process webcam stream with Sobel edge detection.

    Args:
        camera_id: Webcam device ID
        threshold: Edge detection threshold (0.0-1.0). Lower = more forgiving
        blur: Gaussian blur kernel size (odd number). Higher = smoother/more forgiving
    """
    # Initialize filter and source with adjustable parameters
    filter_obj = SobelEdgeFilter(threshold=threshold, blur_kernel_size=blur)
    source = WebcamSource(camera_id)

    # Create and run pipeline
    pipeline = VideoPipeline(source, filter_obj)
    pipeline.process_realtime()


def main():
    # Setup paths
    input_dir = Path("input")
    output_dir = Path("output")

    # Find first MP4 file in input directory
    video_files = list(input_dir.glob("*.mp4"))

    if not video_files:
        print("No MP4 files found in input/ directory")
        print("\nTo use webcam instead, uncomment the following line:")
        print("# process_webcam(camera_id=0)")
        return

    input_video = video_files[0]
    output_video = output_dir / f"edge_{input_video.name}"

    # Process the video file
    # process_video_file(input_video, output_video, show_preview=False)

    # Webcam with Sobel edge detection
    # Adjust threshold (0.0-1.0) - lower = more edges/forgiving
    # Adjust blur (odd number) - higher = smoother edges
    process_webcam(camera_id=0, threshold=0.03, blur=7)


if __name__ == "__main__":
    main()
