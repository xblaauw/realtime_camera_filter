#!/usr/bin/env python3
"""
Real-time webcam processing with various filters.

Usage:
    python webcam.py [camera_id] --filter FILTER [filter-options]

Examples:
    python webcam.py --filter sobel --threshold 0.03
    python webcam.py --filter cartoon --color-levels 6
    python webcam.py --filter thermal --colormap hot
    python webcam.py --filter pixel --pixel-size 16
"""

import argparse
from sources import WebcamSource
from pipeline import VideoPipeline
from filters import list_filters, get_filter_info, create_filter


def build_parser():
    """Build argument parser with dynamically generated filter arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time webcam processing with various filters',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        choices=list_filters(),
        default='sobel',
        help='Filter to apply (default: sobel)'
    )

    parser.add_argument(
        '--virtual-cam',
        type=str,
        help='Output to virtual camera device (e.g., /dev/video2)'
    )

    # Add arguments for all filters
    for filter_name in list_filters():
        filter_info = get_filter_info(filter_name)
        params = filter_info['parameters']

        for param_name, param_info in params.items():
            arg_name = f'--{param_name}'
            arg_kwargs = {
                'help': param_info.get('help', ''),
                'default': param_info.get('default')
            }

            param_type = param_info.get('type')

            if param_type == bool:
                arg_kwargs['action'] = 'store_true'
                arg_kwargs.pop('default', None)
            else:
                arg_kwargs['type'] = param_type

            if 'choices' in param_info:
                arg_kwargs['choices'] = param_info['choices']

            parser.add_argument(arg_name, **arg_kwargs)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Get filter info
    filter_info = get_filter_info(args.filter)
    params = filter_info['parameters']

    # Build filter kwargs from parsed args
    filter_kwargs = {}
    for param_name in params.keys():
        arg_value = getattr(args, param_name.replace('-', '_'), None)
        if arg_value is not None:
            filter_kwargs[param_name.replace('-', '_')] = arg_value

    # Create filter
    filter_obj = create_filter(args.filter, **filter_kwargs)

    # Print filter info
    print(f"Filter: {filter_info['description']}")
    for param_name, value in filter_kwargs.items():
        print(f"  {param_name}: {value}")

    print(f"\nStarting webcam {args.camera_id}")
    print(f"Press 'q' to quit\n")

    # Initialize source and run pipeline
    source = WebcamSource(args.camera_id)
    pipeline = VideoPipeline(source, filter_obj)
    pipeline.process_realtime(virtual_cam_device=args.virtual_cam)


if __name__ == "__main__":
    main()
