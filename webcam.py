#!/usr/bin/env python3
"""
Real-time webcam processing with Sobel edge detection.

Usage:
    python webcam.py [camera_id] [--threshold THRESHOLD] [--blur BLUR]

Examples:
    python webcam.py                          # Default settings
    python webcam.py 0 --threshold 0.03      # Camera 0, low threshold (more edges)
    python webcam.py 1 --threshold 0.1 --blur 5  # Camera 1, higher threshold, blur=5
"""

import sys
import argparse
from pipeline import SobelEdgeFilter, WebcamSource, VideoPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Real-time webcam processing with Sobel edge detection'
    )
    parser.add_argument(
        'camera_id',
        type=int,
        nargs='?',
        default=0,
        help='Webcam device ID (default: 0)'
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

    args = parser.parse_args()

    # Validate blur is odd
    if args.blur % 2 == 0:
        print(f"Warning: blur must be odd, using {args.blur + 1} instead")
        args.blur += 1

    print(f"Starting webcam {args.camera_id} with:")
    print(f"  Threshold: {args.threshold} (lower = more edges)")
    print(f"  Blur: {args.blur} (higher = smoother)")
    print(f"Press 'q' to quit")

    # Initialize filter and source
    filter_obj = SobelEdgeFilter(threshold=args.threshold, blur_kernel_size=args.blur)
    source = WebcamSource(args.camera_id)

    # Create and run pipeline
    pipeline = VideoPipeline(source, filter_obj)
    pipeline.process_realtime()


if __name__ == "__main__":
    main()
