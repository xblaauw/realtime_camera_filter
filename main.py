from pathlib import Path
from pipeline import SobelEdgeFilter, VideoFileSource, WebcamSource, VideoPipeline


def process_video_file(input_path: Path, output_path: Path, show_preview: bool = False):
    """Process a video file with Sobel edge detection."""
    print(f"Loading video from: {input_path}")

    # Initialize filter and source
    filter_obj = SobelEdgeFilter()
    source = VideoFileSource(input_path)

    # Create and run pipeline
    pipeline = VideoPipeline(source, filter_obj)
    pipeline.process_to_file(output_path, show_preview=show_preview)


def process_webcam(camera_id: int = 0):
    """Process webcam stream with Sobel edge detection."""
    # Initialize filter and source
    filter_obj = SobelEdgeFilter()
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
    process_video_file(input_video, output_video, show_preview=False)

    # To use webcam instead, uncomment:
    # process_webcam(camera_id=0)


if __name__ == "__main__":
    main()
