#!/usr/bin/env python3
"""
Real-time webcam processing with various filters.

Usage:
    python webcam.py [camera_id] --filter FILTER [filter-specific options]

Examples:
    python webcam.py --filter sobel --threshold 0.03 --blur 7
    python webcam.py --filter cartoon --edge-threshold 0.2 --color-levels 8
    python webcam.py --filter thermal --colormap jet --contrast 1.5
    python webcam.py --filter pixel --pixel-size 16 --smooth
"""

import argparse
import cv2
from pipeline import (
    SobelEdgeFilter, CartoonFilter, ThermalVisionFilter, PixelationFilter,
    WebcamSource, VideoPipeline
)


def main():
    parser = argparse.ArgumentParser(
        description='Real-time webcam processing with various filters',
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
        'camera_id',
        type=int,
        nargs='?',
        default=0,
        help='Webcam device ID (default: 0)'
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

    args = parser.parse_args()

    # Create the appropriate filter
    if args.filter == 'sobel':
        # Validate blur is odd
        if args.blur % 2 == 0:
            args.blur += 1
        filter_obj = SobelEdgeFilter(threshold=args.threshold, blur_kernel_size=args.blur)
        print(f"Sobel Edge Detection:")
        print(f"  Threshold: {args.threshold}")
        print(f"  Blur: {args.blur}")

    elif args.filter == 'cartoon':
        filter_obj = CartoonFilter(
            edge_threshold=args.edge_threshold,
            color_levels=args.color_levels,
            blur_d=args.blur_d
        )
        print(f"Cartoon Filter:")
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
        print(f"Thermal Vision:")
        print(f"  Colormap: {args.colormap}")
        print(f"  Brightness: {args.brightness}")
        print(f"  Contrast: {args.contrast}")

    elif args.filter == 'pixel':
        filter_obj = PixelationFilter(
            pixel_size=args.pixel_size,
            smooth=args.smooth
        )
        print(f"Pixelation:")
        print(f"  Pixel size: {args.pixel_size}")
        print(f"  Smooth: {args.smooth}")

    print(f"\nStarting webcam {args.camera_id}")
    print(f"Press 'q' to quit\n")

    # Initialize source
    source = WebcamSource(args.camera_id)

    # Create and run pipeline
    pipeline = VideoPipeline(source, filter_obj)
    pipeline.process_realtime()


if __name__ == "__main__":
    main()
