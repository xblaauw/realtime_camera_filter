#!/usr/bin/env python3
"""
Process video files with Sobel edge detection.

Usage:
    python process_input.py INPUT_VIDEO [OUTPUT_VIDEO] [--threshold THRESHOLD] [--blur BLUR] [--preview]

Examples:
    python process_input.py input/video.mp4                              # Auto output name
    python process_input.py input/video.mp4 output/edges.mp4             # Custom output
    python process_input.py input/video.mp4 --threshold 0.05 --blur 5    # Custom settings
    python process_input.py input/video.mp4 --preview                    # Show preview while processing
"""

import sys
import argparse
from pathlib import Path
from pipeline import SobelEdgeFilter, VideoFileSource, VideoPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Process video files with Sobel edge detection'
    )
    parser.add_argument(
        'input_video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        'output_video',
        type=str,
        nargs='?',
        help='Path to output video file (default: output/edge_<input_name>)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.03,
        help='Edge detection threshold (0.0-1.0). Lower = more forgiving/more edges (default: 0.03)'
    )
    parser.add_argument(
        '--blur',
        type=int,
        default=7,
        help='Gaussian blur kernel size (odd number). Higher = smoother edges (default: 7)'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview window during processing'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Do not preserve audio from source video'
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if args.output_video:
        output_path = Path(args.output_video)
    else:
        output_path = Path("output") / f"edge_{input_path.name}"

    # Validate blur is odd
    if args.blur % 2 == 0:
        print(f"Warning: blur must be odd, using {args.blur + 1} instead")
        args.blur += 1

    print(f"Processing video with:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Threshold: {args.threshold} (lower = more edges)")
    print(f"  Blur: {args.blur} (higher = smoother)")
    print(f"  Preview: {args.preview}")
    print(f"  Preserve audio: {not args.no_audio}")
    print()

    # Initialize filter and source
    filter_obj = SobelEdgeFilter(threshold=args.threshold, blur_kernel_size=args.blur)
    source = VideoFileSource(input_path)

    # Create and run pipeline
    pipeline = VideoPipeline(source, filter_obj, source_path=input_path)
    pipeline.process_to_file(
        output_path,
        show_preview=args.preview,
        preserve_audio=not args.no_audio
    )


if __name__ == "__main__":
    main()
