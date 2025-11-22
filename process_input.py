#!/usr/bin/env python3
"""
Process video files with various filters.

Usage:
    python process_input.py INPUT_VIDEO [OUTPUT_VIDEO] --filter FILTER [filter-specific options]

Examples:
    python process_input.py input/video.mp4 --filter sobel --threshold 0.03
    python process_input.py input/video.mp4 --filter cartoon --color-levels 6
    python process_input.py input/video.mp4 --filter thermal --colormap hot
    python process_input.py input/video.mp4 --filter pixel --pixel-size 32
"""

import sys
import argparse
import cv2
from pathlib import Path
from pipeline import (
    SobelEdgeFilter, CartoonFilter, ThermalVisionFilter, PixelationFilter,
    VideoFileSource, VideoPipeline
)


def main():
    parser = argparse.ArgumentParser(
        description='Process video files with various filters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Filter-specific parameters:

  sobel:
    --threshold FLOAT     Edge threshold (0.0-1.0). Lower = more edges (default: 0.03)
    --blur INT            Gaussian blur size (odd number). Higher = smoother (default: 7)

  cartoon:
    --edge-threshold FLOAT   Edge sensitivity (0.0-1.0). Lower = more edges (default: 0.2)
    --color-levels INT       Color quantization levels (4-16). Lower = more cartoony (default: 8)
    --blur-d INT             Bilateral filter diameter (5-15). Higher = smoother (default: 9)

  thermal:
    --colormap STR          Color scheme: jet, hot, inferno, cool, bone (default: jet)
    --brightness FLOAT      Brightness (0.5-2.0). Higher = brighter (default: 1.0)
    --contrast FLOAT        Contrast (0.5-3.0). Higher = more contrast (default: 1.5)

  pixel:
    --pixel-size INT        Block size (4-64). Higher = more pixelated (default: 16)
    --smooth                Use smooth pixel edges instead of hard edges
        """
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
        help='Path to output video file (default: output/<filter>_<input_name>)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        choices=['sobel', 'cartoon', 'thermal', 'pixel'],
        default='sobel',
        help='Filter to apply (default: sobel)'
    )

    # Sobel filter parameters
    parser.add_argument('--threshold', type=float, default=0.03)
    parser.add_argument('--blur', type=int, default=7)

    # Cartoon filter parameters
    parser.add_argument('--edge-threshold', type=float, default=0.2)
    parser.add_argument('--color-levels', type=int, default=8)
    parser.add_argument('--blur-d', type=int, default=9)

    # Thermal filter parameters
    parser.add_argument('--colormap', type=str, default='jet',
                        choices=['jet', 'hot', 'inferno', 'cool', 'bone', 'viridis'])
    parser.add_argument('--brightness', type=float, default=1.0)
    parser.add_argument('--contrast', type=float, default=1.5)

    # Pixelation filter parameters
    parser.add_argument('--pixel-size', type=int, default=16)
    parser.add_argument('--smooth', action='store_true')

    # General options
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
        output_path = Path("output") / f"{args.filter}_{input_path.name}"

    # Create the appropriate filter
    if args.filter == 'sobel':
        # Validate blur is odd
        if args.blur % 2 == 0:
            args.blur += 1
        filter_obj = SobelEdgeFilter(threshold=args.threshold, blur_kernel_size=args.blur)
        print(f"Filter: Sobel Edge Detection")
        print(f"  Threshold: {args.threshold}")
        print(f"  Blur: {args.blur}")

    elif args.filter == 'cartoon':
        filter_obj = CartoonFilter(
            edge_threshold=args.edge_threshold,
            color_levels=args.color_levels,
            blur_d=args.blur_d
        )
        print(f"Filter: Cartoon")
        print(f"  Edge threshold: {args.edge_threshold}")
        print(f"  Color levels: {args.color_levels}")
        print(f"  Blur diameter: {args.blur_d}")

    elif args.filter == 'thermal':
        # Map colormap name to cv2 constant
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'inferno': cv2.COLORMAP_INFERNO,
            'cool': cv2.COLORMAP_COOL,
            'bone': cv2.COLORMAP_BONE,
            'viridis': cv2.COLORMAP_VIRIDIS
        }
        filter_obj = ThermalVisionFilter(
            colormap=colormap_dict[args.colormap],
            brightness=args.brightness,
            contrast=args.contrast
        )
        print(f"Filter: Thermal Vision")
        print(f"  Colormap: {args.colormap}")
        print(f"  Brightness: {args.brightness}")
        print(f"  Contrast: {args.contrast}")

    elif args.filter == 'pixel':
        filter_obj = PixelationFilter(
            pixel_size=args.pixel_size,
            smooth=args.smooth
        )
        print(f"Filter: Pixelation")
        print(f"  Pixel size: {args.pixel_size}")
        print(f"  Smooth: {args.smooth}")

    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"Preview: {args.preview}")
    print(f"Preserve audio: {not args.no_audio}\n")

    # Initialize source
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
